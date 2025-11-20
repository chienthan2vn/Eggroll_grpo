import os
import jax

from huggingface_hub.constants import HF_HOME

jax.config.update("jax_compilation_cache_dir", os.path.join(HF_HOME, "hyperscaleescomp"))
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

import jax.numpy as jnp


import numpy as np
from datasets import load_dataset

import optax

print("loading dataset")
ds = load_dataset("ylecun/mnist")
# ds = load_dataset("uoft-cs/cifar10")
# ds['train'][0]['image']

num_train_points = len(ds['train'])
num_valid_points = len(ds['test'])

training_images = np.array(ds['train']['image']).reshape((num_train_points, -1))
training_labels = np.array(ds['train']['label'])

image_size = training_images.shape[-1]
num_categories = 10
sigma = 0.01
lr = 1.0
batch_size = 32
group_size = 10
num_iters = num_train_points // batch_size

print(training_images.shape, training_labels.shape)


import hyperscalees as hs

MODEL = hs.models.common.MLP

key = jax.random.key(0)
model_key = jax.random.fold_in(key, 0)
es_key = jax.random.fold_in(key, 1)

frozen_params, params, scan_map, es_map = MODEL.rand_init(model_key, in_dim=image_size, out_dim=num_categories, hidden_dims=[1024, 1024], use_bias=False, activation="silu", dtype="float32")
es_tree_key = hs.models.common.simple_es_tree_key(params, es_key, scan_map)


NOISER = hs.noiser.eggroll.EggRoll
# NOISER = hs.noiser.open_es.OpenES
# frozen_noiser_params, noiser_params = NOISER.init_noiser(params, sigma, lr, group_size=group_size, solver=optax.sgd)
frozen_noiser_params, noiser_params = NOISER.init_noiser(params, sigma, optax.linear_schedule(init_value=lr, end_value=0.0, transition_steps=num_iters), group_size=group_size, solver=optax.sgd)

# inputs are noiser_params, params, iterinfo, input
jit_forward = jax.jit(jax.vmap(lambda n, p, i, x: MODEL.forward(NOISER, frozen_noiser_params, n, frozen_params, p, es_tree_key, i, x), in_axes=(None, None, 0, 0)))
# inputs are noiser_params, params, input
jit_forward_eval = jax.jit(jax.vmap(lambda n, p, x: MODEL.forward(NOISER, frozen_noiser_params, n, frozen_params, p, es_tree_key, None, x), in_axes=(None, None, 0)))
# inputs are noiser_params, params, fitnesses, iterinfo
jit_update = jax.jit(lambda n, p, f, i: NOISER.do_updates(frozen_noiser_params, n, p, es_tree_key, f, i, es_map))


# ema = None
last_accuracies = []
for epoch in range(num_iters):
    image_batch = training_images[epoch * batch_size: (epoch + 1) * batch_size]
    answers_batch = training_labels[epoch * batch_size: (epoch + 1) * batch_size]

    iterinfo = (jnp.full(batch_size * group_size, epoch, dtype=jnp.int32), jnp.arange(batch_size * group_size))

    answer_logits = jit_forward(noiser_params, params, iterinfo, jnp.repeat(image_batch, group_size, axis=0))
    # print(answer_logits.shape)

    # calculating fitness
    chosen_answer = jnp.argmax(answer_logits, axis=-1)
    is_correct = (chosen_answer == jnp.repeat(answers_batch, group_size, axis=0)).astype(jnp.float32)
    raw_score = -optax.losses.softmax_cross_entropy_with_integer_labels(answer_logits, jnp.repeat(answers_batch, group_size, axis=0))
    # raw_score = is_correct
    # print("accuracy:", jnp.mean(is_correct))
    accuracy = jnp.mean(is_correct)

    fitnesses = NOISER.convert_fitnesses(frozen_noiser_params, noiser_params, raw_score)
    noiser_params, params = jit_update(noiser_params, params, fitnesses, iterinfo)

    # if ema is None:
    #     ema = accuracy
    # else:
    #     ema = 0.9 * ema + 0.1 * accuracy

    last_accuracies.append(accuracy)

    if epoch % 100 == 0:
        print(f"accuracy: {accuracy:.2f}; mean past accuracies: {sum(last_accuracies) / len(last_accuracies):.2f}; (length is {len(last_accuracies)})")
        last_accuracies = []

    # noiser_params["sigma"] = sigma * (1 - epoch / num_iters)

print("getting validation performance")
validation_images = np.array(ds['test']['image']).reshape((num_valid_points, -1))
validation_labels = np.array(ds['test']['label'])

answer_logits = jit_forward_eval(noiser_params, params, validation_images)
chosen_answer = jnp.argmax(answer_logits, axis=-1)
is_correct = (chosen_answer == validation_labels).astype(jnp.float32)
print(jnp.mean(is_correct))
