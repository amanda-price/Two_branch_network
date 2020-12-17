from src import *
import tensorflow as tf
import numpy as np

batch_size = 1024
epochs = 10
image_feat_path = "./data/test_data/flickr30k/flickr30k_vgg_image_feat_test.mat"
sent_feat_path = "./data/test_data/flickr30k/flickr30k_text_feat_test.mat"
shuffle_step=5

data_loader = DatasetLoader(image_feat_path, sent_feat_path)

model = embedding_model(data_loader.im_feats.shape[1:], data_loader.sent_feats.shape[1:])
loss = EmbeddingLoss(1, batch_size)

sent_sample_size = 5 # how many sentences per image

total_loss, batch_loss = [], []
optimizer = tf.keras.optimizers.Adam()

def _get_data(data_loader):
    img_feats = data_loader.im_feats
    sent_feats = data_loader.sent_feats

    def gen():
        for ind, img in enumerate(img_feats):
            for i in range(0,5):
                yield img, sent_feats[ind*5+i]
    return gen

data = (
    tf.data.Dataset
    .from_generator(_get_data(data_loader), (tf.float32, tf.float32))
    .shuffle(1000)
    .batch(batch_size)
)

optimizer = tf.keras.optimizers.Adam()
for epoch in range(10):
    for im_feats, sent_feats in data:
        _batch_loss = tf.constant(0, dtype=tf.float32)
        with tf.GradientTape() as tape:
            labels = np.repeat(np.eye(im_feats.shape[0], dtype=bool), 1, axis=0)
            [im_emb, sent_emb] = model([im_feats, sent_feats])

            _batch_loss += loss(im_emb, sent_emb, labels)

        trainable_variables = model.trainable_variables

        gradients = tape.gradient(_batch_loss, trainable_variables)
        optimizer.apply_gradients(zip(gradients, trainable_variables))

        total_loss.append(_batch_loss / int(im_feats.shape[1]))