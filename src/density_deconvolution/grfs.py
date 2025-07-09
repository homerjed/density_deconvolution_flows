
import os 
from pathlib import Path
from functools import partial
from copy import deepcopy
from typing import Optional, Callable, Union, Literal

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax
from jaxtyping import Array, PRNGKeyArray, Float, Int, Bool, Scalar, PyTree, jaxtyped
from beartype import beartype as typechecker

from einops import rearrange
import matplotlib.pyplot as plt
from tqdm.auto import trange

os.environ["TYPECHECK"] = "0"

from transformer_flows import (
    TransformerFlow, CausalTransformerBlock, Permutation, 
    Policy, Dataset,
    get_shardings, exists, clear_and_get_results_dir,
    add_noise, apply_ema,
    save, load, NoiseType, count_parameters, loader,
    shard_model, shard_batch
)


def generate_grf(
    size: int,
    power_spectrum_fn: Callable[[Array], Array],
    key: PRNGKeyArray,
    dim: int = 2
) -> Array:
    freqs = jnp.fft.fftfreq(size) * size

    shape = (size,) * dim
    if dim == 1:
        k_mag = jnp.abs(freqs)
    elif dim == 2:
        kx, ky = jnp.meshgrid(freqs, freqs, indexing='ij')
        k_mag = jnp.sqrt(kx ** 2. + ky ** 2.)
    elif dim == 3:
        kx, ky, kz = jnp.meshgrid(freqs, freqs, freqs, indexing='ij')
        k_mag = jnp.sqrt(kx ** 2. + ky ** 2. + kz ** 2.)

    Pk_sqrt = jnp.sqrt(power_spectrum_fn(k_mag))

    def sample_grf(key: PRNGKeyArray) -> Array:
        real, imag = jr.normal(key, (2,) + shape)
        return jnp.fft.ifftn((real + 1j * imag) * Pk_sqrt / jnp.sqrt(2.)).real
    
    return sample_grf(key)


key = jr.key(0)

dataset_name = "grfs"

n_pix = 32

noise_covariance = jnp.eye(n_pix) * 0.01


def data_likelihood(w, v):
    return jax.scipy.stats.multivariate_normal.logpdf(w, mean=v, cov=noise_covariance)


def power_spectrum(k: Array) -> Array: 
    return 1. / (1. + k ** 2.)


def generator(key): 
    return generate_grf(
        size=n_pix, power_spectrum_fn=power_spectrum, key=key, dim=1
    )


def measure(key, v):
    return jr.multivariate_normal(key, mean=v, cov=noise_covariance)


eps_sigma = 0.

model_config_dict = dict(
    img_size=n_pix,
    n_channels=1,
    patch_size=4,
    channels=128,
    n_blocks=3,
    layers_per_block=2,
    head_dim=64,
    expansion=2,
    eps_sigma=eps_sigma
)

n_data = 40_000

keys = jr.split(key, n_data)
data = jax.vmap(generator)(keys)

a, b = jnp.min(data), jnp.max(data)
V = 2. * (data - a) / (b - a) - 1.
W = jax.vmap(measure)(keys, V) # NOTE: reusing keys, what about post process?

def postprocess_fn(x: Array) -> Array: 
    return jnp.clip((1. + x) * 0.5 * (b - a) + a, min=0., max=1.)

target_fn = lambda *args, **kwargs: None

# Matched latents and measurements
v_train, v_valid = jnp.split(V, [int(0.9 * n_data)])
w_train, w_valid = jnp.split(W, [int(0.9 * n_data)])

w_dataset = Dataset(
    dataset_name,
    x_train=w_train, 
    y_train=None, 
    x_valid=w_valid, 
    y_valid=None, 
    target_fn=target_fn, 
    postprocess_fn=postprocess_fn
)


class TransformerFlow1D(TransformerFlow):
    def __init__(
        self, 
        *args, 
        conditioning_type: Optional[Literal["layernorm", "embed"]] = None, 
        y_dim: Optional[int] = None, 
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.n_patches = int(kwargs["img_size"] / kwargs["patch_size"])
        self.sequence_dim = kwargs["n_channels"] * kwargs["patch_size"]

        def _make_block(permute: Int[Array, ""], key: PRNGKeyArray) -> CausalTransformerBlock:
            block = CausalTransformerBlock(
                self.sequence_dim,
                kwargs["channels"],
                n_patches=self.n_patches,
                permutation=Permutation(
                    permute=permute,
                    sequence_length=self.n_patches
                ), 
                n_layers=kwargs["layers_per_block"],
                patch_size=kwargs["patch_size"],
                head_dim=kwargs["head_dim"],
                expansion=kwargs["expansion"],
                y_dim=y_dim,
                conditioning_type=conditioning_type,
                key=key
            )
            return block 

        block_keys = jr.split(key, kwargs["n_blocks"])
        permutes = jnp.arange(kwargs["n_blocks"]) % 2 # Alternate permutations
        self.blocks = eqx.filter_vmap(_make_block)(permutes, block_keys)

        self.eps_sigma = kwargs["eps_sigma"]

        if exists(self.eps_sigma):
            assert self.eps_sigma >= 0., (
                "Noise sigma must be positive or zero."
            )

    def patchify(self, x: Float[Array, "d"]) -> Float[Array, "s p"]:
        s = int(self.img_size / self.patch_size)
        p = self.patch_size
        return rearrange(x, "(s p) -> s p", s=s, p=p)

    def unpatchify(self, x):
        s = int(self.img_size / self.patch_size)
        p = self.patch_size
        return rearrange(x, "s p -> (s p)", s=s, p=p)

    # def log_prob(self, x, y=None):
    #     # Cast this calculation to float32
    #     ...


imgs_dir = clear_and_get_results_dir(dataset_name)

key_model, key_train, key_data = jr.split(key, 3)

sharding, replicated_sharding = get_shardings()

policy = Policy(
    compute_dtype=jnp.float32, # NOTE: likelihood calculation float32
    param_dtype=jnp.float32,
    output_dtype=jnp.float32
)

key_p, key_q = jr.split(key)

p_model, p_state = eqx.nn.make_with_state(TransformerFlow1D)(
    **model_config_dict, key=key_p
)

q_model, q_state = eqx.nn.make_with_state(TransformerFlow1D)(
    y_dim=W.shape[-1], conditioning_type="layernorm", **model_config_dict, key=key_q
)


class DeconvolutionModel(eqx.Module):
    p_model: TransformerFlow1D
    q_model: TransformerFlow1D

    def __init__(
        self, 
        p_model: TransformerFlow1D, 
        q_model: TransformerFlow1D
    ):
        self.p_model = p_model
        self.q_model = q_model

    def sample_latents(*args, **kwargs):
        return p_model.sample_model(*args, **kwargs)


model = DeconvolutionModel(p_model, q_model)

train_config_dict = dict(
    eps_sigma=eps_sigma,
    noise_type="gaussian",
    batch_size=100,
    K=100,
    n_epochs=1000,
    lr=1e-3,
    n_epochs_warmup=10,
    initial_lr=1e-6,
    final_lr=1e-6,
    max_grad_norm=0.5,
    use_ema=False,
    ema_rate=0.9999,
    accumulate_gradients=False,
    n_minibatches=0,
    sample_every=1000,
    denoise_samples=True,
    n_sample=10,
    n_warps=5,
    cmap="PuOr",
    policy=policy,
    imgs_dir=imgs_dir
)

x_fixed_ = jax.vmap(w_dataset.postprocess_fn)(V[:train_config_dict["n_sample"] ** 2])

plt.figure(dpi=200)
plt.imshow(x_fixed_, cmap=train_config_dict["cmap"]) 
plt.xticks([])
plt.yticks([])
plt.colorbar()
plt.savefig(imgs_dir / "data_v.png", bbox_inches="tight")
plt.close()


@eqx.filter_jit
def sample_model(
    model: TransformerFlow1D, 
    z: Float[Array, "#n s q"], 
    y: Optional[Float[Array, "..."]],  # NOTE: NOT batched like z
    state: eqx.nn.State,
    *,
    guidance: float = 0.,
    attention_temperature: float = 1.,
    guide_what: Optional[Literal["ab", "a", "b"]] = "ab",
    return_sequence: bool = False,
    denoise_samples: bool = False,
    sharding: Optional[jax.sharding.NamedSharding] = None,
    replicated_sharding: Optional[jax.sharding.NamedSharding] = None
) -> Union[
    Float[Array, "#n c h w"], Float[Array, "#n t c h w"]
]:
    model = shard_model(model, sharding=replicated_sharding)
    z, y = shard_batch((z, y), sharding=sharding)

    # Sample
    sample_fn = lambda z, y: model.reverse(
        z, 
        y, 
        state=state, 
        guidance=guidance,
        guide_what=guide_what,
        attention_temperature=attention_temperature,
        return_sequence=return_sequence
    )

    if exists(y):
        samples, state = sample_fn(z, y) # Sampling q(v|w)
    else:
        samples, state = eqx.filter_vmap(sample_fn)(z, y) # Sampling otherwise...

    # Denoising
    if denoise_samples:
        if return_sequence:
            denoised = jax.vmap(model.denoise)(samples[:, -1], y)
            samples = jnp.concatenate(
                [samples, denoised[:, jnp.newaxis]], axis=1
            )
        else:
            samples = jax.vmap(model.denoise)(samples, y)

    return samples


def ELBO(
    key: PRNGKeyArray,
    model: DeconvolutionModel,
    w: Array,
    *,
    q_state: eqx.nn.State,
    K: int = 10, 
    policy: Optional[Policy] = None
) -> tuple[Scalar, dict[str, Array]]:

    if exists(policy):
        w = policy.cast_to_compute(w)
        model = policy.cast_to_compute(model)

    def elbo(v, w):
        log_prob_v = model.p_model.log_prob(v, y=None) 
        log_prob_v_w = model.q_model.log_prob(v, y=w)
        log_p_n = data_likelihood(w, v)
        return log_p_n + log_prob_v - log_prob_v_w, (log_prob_v, log_prob_v_w)

    def w_i_sampler(key, w): 
        def _sampler_fn(z): 
            return sample_model(model.q_model, z, y=w, state=q_state)
        z = q_model.sample_prior(key_prior, K)
        return eqx.filter_vmap(_sampler_fn)(z)

    key_prior, key_sample = jr.split(key)

    # Sample K samples per input datapoint w_i
    keys = jr.split(key_sample, w.shape[0])
    v = eqx.filter_vmap(w_i_sampler)(keys, w) # (len(w), K, data_dim)

    # Vmap over K-axis then batch axes e.g. (v_K)_i and w_i
    outs = eqx.filter_vmap(lambda v: eqx.filter_vmap(elbo)(v, w), in_axes=1)(v)

    elbo, (loss_p, loss_q) = jax.tree.map(jnp.mean, outs)

    if exists(policy):
        elbo = policy.cast_to_output(elbo)

    return elbo, dict(loss_p=loss_p, loss_q=loss_q)


@eqx.filter_jit(donate="all-except-first")
def evaluate(
    model: TransformerFlow, 
    key: PRNGKeyArray, 
    x: Float[Array, "n _ _ _"], 
    y: Optional[Float[Array, "n ..."]] = None,
    *,
    K: int = 10,
    q_state: eqx.nn.State,
    policy: Optional[Policy] = None,
    sharding: Optional[jax.sharding.NamedSharding] = None,
    replicated_sharding: Optional[jax.sharding.NamedSharding] = None
) -> tuple[Scalar, dict[str, Array]]:

    model = shard_model(model, sharding=replicated_sharding)

    x, y = shard_batch((x, y), sharding=sharding)

    loss, metrics = ELBO(key, model, x, q_state=q_state, policy=policy)

    return loss, metrics


def accumulate_gradients_scan(
    model: eqx.Module,
    key: PRNGKeyArray,
    x: Float[Array, "n _ _ _"], 
    n_minibatches: int,
    *,
    grad_fn: Callable[
        [
            eqx.Module, 
            PRNGKeyArray,
            Float[Array, "n _ _ _"],
            Optional[Float[Array, "n ..."]]
        ],
        tuple[Scalar, dict[str, Array]]
    ]
) -> tuple[tuple[Scalar, dict[str, Array]], PyTree]:

    batch_size = x.shape[0]
    minibatch_size = int(batch_size / n_minibatches)

    keys = jr.split(key, n_minibatches)

    def _minibatch_step(minibatch_idx):
        # Gradients and metrics for a single minibatch

        slicer = lambda x: jax.lax.dynamic_slice_in_dim(  
            x, 
            start_index=minibatch_idx * minibatch_size, 
            slice_size=minibatch_size, 
            axis=0
        )
        _x = jax.tree.map(slicer, x)

        (step_L, step_metrics), step_grads = grad_fn(
            keys[minibatch_idx], model, _x
        )

        return step_grads, step_L, step_metrics

    def _scan_step(carry, minibatch_idx):
        # Scan step function for looping over minibatches
        step_grads, step_L, step_metrics = _minibatch_step(minibatch_idx)
        carry = jax.tree.map(jnp.add, carry, (step_grads, step_L, step_metrics))
        return carry, None

    def _get_grads_loss_metrics_shapes():
        # Determine initial shapes for gradients and metrics.
        grads_shapes, L_shape, metrics_shape = jax.eval_shape(_minibatch_step, 0)
        grads = jax.tree.map(lambda x: jnp.zeros(x.shape, x.dtype), grads_shapes)
        L = jax.tree.map(lambda x: jnp.zeros(x.shape, x.dtype), L_shape)
        metrics = jax.tree.map(lambda x: jnp.zeros(x.shape, x.dtype), metrics_shape)
        return grads, L, metrics

    grads, L, metrics = _get_grads_loss_metrics_shapes()
        
    (grads, L, metrics), _ = jax.lax.scan(
        _scan_step, 
        init=(grads, L, metrics), 
        xs=jnp.arange(n_minibatches), 
        length=n_minibatches
    )

    grads = jax.tree.map(lambda g: g / n_minibatches, grads)
    metrics = jax.tree.map(lambda m: m / n_minibatches, metrics)

    return (L / n_minibatches, metrics), grads # Same signature as unaccumulated 


@eqx.filter_jit(donate="all")
def make_step(
    model: DeconvolutionModel, 
    x: Float[Array, "n _ _ _"], 
    y: Optional[Float[Array, "n ..."]], # Arbitrary conditioning shape is flattened
    key: PRNGKeyArray, 
    opt_state: optax.OptState, 
    opt: optax.GradientTransformation,
    *,
    K: int = 10,
    q_state: eqx.nn.State,
    n_minibatches: Optional[int] = 4,
    accumulate_gradients: Optional[bool] = False,
    policy: Optional[Policy] = None,
    sharding: Optional[jax.sharding.NamedSharding] = None,
    replicated_sharding: Optional[jax.sharding.NamedSharding] = None
) -> tuple[
    Scalar, dict[str, Array], TransformerFlow1D, optax.OptState
]:

    model, opt_state = shard_model(model, opt_state, replicated_sharding)
    x = shard_batch(x, sharding)

    grad_fn = eqx.filter_value_and_grad(
        partial(ELBO, policy=policy, q_state=q_state, K=K), has_aux=True
    )

    if exists(policy):
        model = policy.cast_to_compute(model)

    if accumulate_gradients and n_minibatches:
        (loss, metrics), grads = accumulate_gradients_scan(
            model, 
            key, 
            x, 
            n_minibatches=n_minibatches, 
            grad_fn=grad_fn
        ) 
    else:
        (loss, metrics), grads = grad_fn(key, model, x)

    if exists(policy):
        grads = policy.cast_to_param(grads)
        model = policy.cast_to_param(model)

    updates, opt_state = opt.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)

    return loss, metrics, model, opt_state


def train(
    key: PRNGKeyArray,
    # Data
    dataset: Dataset, 
    # Model
    model: DeconvolutionModel,
    q_state: eqx.nn.State,
    eps_sigma: Optional[float],
    noise_type: NoiseType,
    # Training
    K: int = 10,
    batch_size: int = 256, 
    n_epochs: int = 100,
    lr: float = 2e-4,
    n_epochs_warmup: int = 1, # Cosine decay schedule 
    initial_lr: float = 1e-6, # Cosine decay schedule
    final_lr: float = 1e-6, # Cosine decay schedule
    max_grad_norm: Optional[float] = 1.0,
    use_ema: bool = False,
    ema_rate: Optional[float] = 0.9995,
    accumulate_gradients: bool = False,
    n_minibatches: int = 4,
    policy: Optional[Policy] = None,
    # Sampling
    sample_every: int = 1000,
    n_sample: Optional[int] = 4,
    n_warps: Optional[int] = 1,
    denoise_samples: bool = False,
    cmap: Optional[str] = None,
    # Sharding: data and model
    sharding: Optional[jax.sharding.NamedSharding] = None,
    replicated_sharding: Optional[jax.sharding.NamedSharding] = None,
    save_fn: Callable[[Optional[str], TransformerFlow], None] = None,
    imgs_dir: str | Path = Path.cwd() / "imgs"
) -> TransformerFlow:

    print("n_params_p={:.3E}".format(count_parameters(model)))

    valid_key, sample_key, *loader_keys = jr.split(key, 4)

    # Optimiser & scheduler
    n_steps_per_epoch = int((dataset.x_train.shape[0] + dataset.x_valid.shape[0]) / batch_size) 
    n_steps = n_epochs * n_steps_per_epoch

    scheduler = optax.warmup_cosine_decay_schedule(
        init_value=initial_lr, 
        peak_value=lr, 
        warmup_steps=n_epochs_warmup * n_steps_per_epoch,
        decay_steps=n_epochs * n_steps_per_epoch, 
        end_value=final_lr
    )

    opt = optax.adamw(
        learning_rate=scheduler, b1=0.9, b2=0.95, weight_decay=1e-4
    )
    if exists(max_grad_norm):
        assert max_grad_norm > 0.
        opt = optax.chain(optax.clip_by_global_norm(max_grad_norm), opt)

    opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))

    if use_ema:
        ema_model = deepcopy(model) 

    _batch_size = n_minibatches * batch_size if accumulate_gradients else batch_size

    losses, metrics = [], []
    with trange(n_steps) as bar: 
        for i, (x_t, y_t), (x_v, y_v) in zip(
            bar, 
            loader(dataset.x_train, dataset.y_train, _batch_size, key=loader_keys[0]), 
            loader(dataset.x_valid, dataset.y_valid, _batch_size, key=loader_keys[1])
        ):
            key_eps, key_step = jr.split(jr.fold_in(key, i))

            # Train 
            loss_t, metrics_t, model, opt_state = make_step(
                model, 
                add_noise(
                    x_t, key_eps, noise_type=noise_type, eps_sigma=eps_sigma
                ), 
                y_t, 
                key_step, 
                opt_state, 
                opt, 
                q_state=q_state,
                K=K,
                n_minibatches=n_minibatches,
                accumulate_gradients=accumulate_gradients,
                policy=policy,
                sharding=sharding,
                replicated_sharding=replicated_sharding
            )

            if use_ema:
                ema_model = apply_ema(
                    ema_model, model, ema_rate=ema_rate, policy=policy
                )

            # Validate
            loss_v, metrics_v = evaluate(
                ema_model if use_ema else model, 
                valid_key, 
                add_noise(
                    x_v, key_eps, noise_type=noise_type, eps_sigma=eps_sigma
                ), 
                y_v, 
                K=K,
                q_state=q_state,
                policy=policy,
                sharding=sharding,
                replicated_sharding=replicated_sharding
            )

            # Record
            losses.append((loss_t, loss_v))
            metrics.append(
                (
                    (metrics_t["loss_p"], metrics_v["loss_q"]), 
                    (metrics_t["loss_p"], metrics_v["loss_q"])
                )
            )

            bar.set_postfix_str("Lt={:.3E} Lv={:.3E}".format(loss_t, loss_v))

            # Sample
            if (i % sample_every == 0) or (i in [10, 100, 500]):

                # Plot training data 
                if (i == 0) and exists(n_sample):
                    x_fixed = x_t[:n_sample ** 2] # Fix first batch
                    # y_fixed = y_t[:n_sample ** 2] if use_y else None

                    x_fixed_ = jax.vmap(dataset.postprocess_fn)(x_fixed)

                    plt.figure(dpi=200)
                    plt.imshow(x_fixed_, cmap=cmap) 
                    plt.xticks([])
                    plt.yticks([])
                    plt.colorbar()
                    plt.savefig(imgs_dir / "data_w.png", bbox_inches="tight")
                    plt.close()

                if exists(n_sample):
                    z = model.p_model.sample_prior(sample_key, n_sample ** 2) 

                    samples = sample_model(
                        (ema_model if use_ema else model).p_model, 
                        z, 
                        y=None, 
                        state=q_state,
                        # guidance=guidance,
                        denoise_samples=denoise_samples,
                        sharding=sharding,
                        replicated_sharding=replicated_sharding
                    )

                    samples = jax.vmap(dataset.postprocess_fn)(samples)

                    plt.figure(dpi=200)
                    plt.imshow(samples, cmap=cmap) 
                    plt.xticks([])
                    plt.yticks([])
                    plt.colorbar()
                    plt.savefig(imgs_dir / "samples/samples_{:05d}.png".format(i), bbox_inches="tight")
                    plt.close() 
                    
                # Losses and metrics
                if i > 0:

                    def filter_spikes(l: list, loss_max: float = 10.0) -> list[float]:
                        return [float(_l) for _l in l if _l < loss_max]

                    fig, axs = plt.subplots(1, 3, figsize=(11., 3.))
                    ax = axs[0]
                    ax.plot(filter_spikes([l for l, _ in losses]), label="train") 
                    ax.plot(filter_spikes([l for _, l in losses]), label="valid [ema]" if use_ema else "valid") 
                    ax.set_title(r"$L$")
                    ax.legend(frameon=False)
                    ax = axs[1]
                    ax.plot(filter_spikes([m[0][0] for m in metrics]))
                    ax.plot(filter_spikes([m[0][1] for m in metrics]))
                    ax.axhline(1., linestyle=":", color="k")
                    ax.set_title(r"$z^2$")
                    ax = axs[2]
                    ax.plot(filter_spikes([m[1][0] for m in metrics]))
                    ax.plot(filter_spikes([m[1][1] for m in metrics]))
                    ax.set_title(r"$\sum_t^T\log|\mathbf{J}_t|$")
                    for ax in axs:
                        ax.set_xscale("log")
                    plt.savefig(imgs_dir / "losses.png", bbox_inches="tight")
                    plt.close()

                if exists(save_fn):
                    save_fn(model=ema_model if use_ema else model)

    return model


model = train(
    key_train,
    w_dataset,
    model,
    q_state,
    **train_config_dict
)