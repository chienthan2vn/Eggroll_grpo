import jax
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
sigma = 0.0
lr = 1e-3
batch_size = 32
group_size = 1
num_iters = num_train_points // batch_size

print(training_images.shape, training_labels.shape)


import hyperscalees as hs

MODEL = hs.models.common.MLP

key = jax.random.key(0)
model_key = jax.random.fold_in(key, 0)
es_key = jax.random.fold_in(key, 1)

frozen_params, params, scan_map, es_map = MODEL.rand_init(model_key, in_dim=image_size, out_dim=num_categories, hidden_dims=[1024, 1024], use_bias=False, activation="silu", dtype="float32")
es_tree_key = hs.models.common.simple_es_tree_key(params, es_key, scan_map)


NOISER = hs.noiser.base_noiser.Noiser
frozen_noiser_params, noiser_params = NOISER.init_noiser(params, sigma, lr)

# inputs are noiser_params, params, iterinfo, input
jit_forward = jax.jit(jax.vmap(lambda n, p, i, x: MODEL.forward(NOISER, frozen_noiser_params, n, frozen_params, p, es_tree_key, i, x), in_axes=(None, None, 0, 0)))
# inputs are noiser_params, params, input
jit_forward_eval = jax.jit(jax.vmap(lambda n, p, x: MODEL.forward(NOISER, frozen_noiser_params, n, frozen_params, p, es_tree_key, None, x), in_axes=(None, None, 0)))
# inputs are noiser_params, params, fitnesses, iterinfo
jit_update = jax.jit(lambda n, p, f, i: NOISER.do_updates(frozen_noiser_params, n, p, es_tree_key, f, i, es_map))

def loss_fn(params, noiser_params, iterinfo, image_batch, answers_batch):
    answer_logits = jit_forward(noiser_params, params, iterinfo, jnp.repeat(image_batch, group_size, axis=0))
    chosen_answer = jnp.argmax(answer_logits, axis=-1)
    is_correct = (chosen_answer == jnp.repeat(answers_batch, group_size, axis=0)).astype(jnp.float32)
    raw_score = optax.losses.softmax_cross_entropy_with_integer_labels(answer_logits, jnp.repeat(answers_batch, group_size, axis=0))

    return jnp.mean(raw_score), is_correct

get_gradient = jax.value_and_grad(loss_fn, has_aux=True)

solver = optax.sgd(learning_rate=lr)
# solver = optax.contrib.prodigy()
opt_state = solver.init(params)


# ema = None
last_accuracies = []
for mega_epoch in range(1):
    print("mega epoch", mega_epoch)
    for epoch in range(num_iters):
        image_batch = training_images[epoch * batch_size: (epoch + 1) * batch_size]
        answers_batch = training_labels[epoch * batch_size: (epoch + 1) * batch_size]

        iterinfo = (jnp.full(batch_size * group_size, epoch, dtype=jnp.int32), jnp.arange(batch_size * group_size))

        (_, is_correct), gradient = get_gradient(params, noiser_params, iterinfo, image_batch, answers_batch)

        updates, opt_state = solver.update(gradient, opt_state, params)
        params = optax.apply_updates(params, updates)
        
        accuracy = jnp.mean(is_correct)

        last_accuracies.append(accuracy)
            
        if epoch % 100 == 0:
            print(f"accuracy: {accuracy:.2f}; mean past accuracies: {sum(last_accuracies) / len(last_accuracies):.2f}; (length is {len(last_accuracies)})")
            last_accuracies = []

print("getting validation performance")
validation_images = np.array(ds['test']['image']).reshape((num_valid_points, -1))
validation_labels = np.array(ds['test']['label'])

answer_logits = jit_forward_eval(noiser_params, params, validation_images)
chosen_answer = jnp.argmax(answer_logits, axis=-1)
is_correct = (chosen_answer == validation_labels).astype(jnp.float32)
print(jnp.mean(is_correct))
