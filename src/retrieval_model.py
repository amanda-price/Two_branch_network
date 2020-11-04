from os import name
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout, ReLU
from tensorflow.keras.models import Model


def fully_connected(name="fc", hidden=2048, dropout=0.25):
    """Fully connected dense layer with batchnorm, relu activateion and dropout

    Args:
        name (str, optional): Name prefix for all layers. Defaults to "fc".
        hidden (int, optional): Hidden node size. Defaults to 2048.
        dropout (float, optional): Dropout fraction. Defaults to 0.25.

    Returns:
        Function pointer to a layer function emulating a keras.layer object
    """
    def _inner(in_layer):
        x = Dense(hidden, name=f"{name}-d1")(in_layer)
        x = BatchNormalization(name=f"{name}-bc")(x)
        x = ReLU(name=f"{name}-relu")(x)
        x = Dropout(dropout, name=f"{name}-d2")(x)
        return x
    return _inner


def embedding_model(img_feats: tuple, sent_feats: tuple) -> Model:
    """Builds a two branch network embedding model

    Args:
        img_feats (tuple): Shape of image features
        sent_feats (tuple): Shape of sentence features

    Returns:
        Model: Embedding model
    """
    img_in = Input(shape=img_feats, name="image_input")
    sent_in = Input(shape=sent_feats, name="sentence_input")

    img_fc = fully_connected(name="img_fc")(img_in)
    img_fc2 = Dense(512, name="fc2-d")(img_fc)
    img_embedded = tf.nn.l2_normalize(img_fc2, 1, 1e-10, name='img-l2')

    sent_fc = fully_connected(name="sent_fc")(sent_in)
    sent_fc2 = Dense(512, name="fc2-sent")(sent_fc)
    sent_embedded = tf.nn.l2_normalize(sent_fc2, 1, 1e-10, name='sent-l2')

    model = Model(inputs=[img_in, sent_in], outputs=[
                  img_embedded, sent_embedded])
    return model


def pdist(x1: tf.Tensor, x2: tf.Tensor) -> tf.Tensor:
    """Calcuates pairwise distance

    Args:
        x1 (tf.Tensor): Tensor of shape (h1, w)
        x2 (tf.Tensor): Tensor of shape (h2, w)

    Returns:
        tf.Tensor: Return pairwise distance for each row vector in x1, x2 as
        a Tensor of shape (h1, h2)
    """

    x1_square = tf.reshape(tf.reduce_sum(x1*x1, axis=1), [-1, 1])
    x2_square = tf.reshape(tf.reduce_sum(x2*x2, axis=1), [1, -1])
    return tf.sqrt(x1_square - 2 * tf.matmul(x1, tf.transpose(x2)) + x2_square + 1e-4)


class EmbeddingLoss:
    def __init__(self, sample_size=None, batch_size=None, margin=None,
                 num_neg_sample=None, sent_only_loss_factor=None, im_loss_factor=None):
        """Calculates embedding loss based on the parameters.

        Args:
            sample_size ([type], optional): [description]. Defaults to None.
            batch_size ([type], optional): [description]. Defaults to None.
            margin ([type], optional): [description]. Defaults to None.
            num_neg_sample ([type], optional): [description]. Defaults to None.
            sent_only_loss_factor ([type], optional): [description]. Defaults to None.
            im_loss_factor ([type], optional): [description]. Defaults to None.

        """
        self.sample_size = sample_size
        self.batch_size = batch_size
        self.margin = margin
        self.num_neg_sample = num_neg_sample
        self.sent_only_loss_factor = sent_only_loss_factor
        self.im_loss_factor = im_loss_factor

    def img_loss(self, sent_im_dist, im_labels, num_sent):
        pos_pair_dist = tf.reshape(tf.boolean_mask(
            sent_im_dist, im_labels), [num_sent, 1])
        neg_pair_dist = tf.reshape(tf.boolean_mask(
            sent_im_dist, ~im_labels), [num_sent, -1])
        im_loss = tf.clip_by_value(
            self.margin + pos_pair_dist - neg_pair_dist, 0, 1e6)
        im_loss = tf.reduce_mean(tf.nn.top_k(
            im_loss, k=self.num_neg_sample)[0])
        return im_loss, pos_pair_dist

    def sent_loss(self):
        pass

    def sent_only_loss(self):
        pass

    def __call__(self, im_embeds: tf.Tensor, sent_embeds: tf.Tensor, im_labels: tf.Tensor) -> tf.Tensor:
        """[summary]

        Args:
            im_embeds (tf.Tensor): [description]
            sent_embeds (tf.Tensor): [description]
            img_lables (tf.Tensor): [description]

        Returns:
            tf.Tensor: [description]
        """
        sent_im_ratio = self.sample_size
        num_img = self.batch_size
        num_sent = num_img * sent_im_ratio

        sent_im_dist = pdist(sent_embeds, im_embeds)
        # image loss: sentence, positive image, and negative image
        im_loss, pos_pair_dist = self.img_loss(sent_im_dist, im_labels, num_sent)

        # sentence loss: image, positive sentence, and negative sentence
        neg_pair_dist = tf.reshape(tf.boolean_mask(tf.transpose(
            sent_im_dist), ~tf.transpose(im_labels)), [num_img, -1])
        neg_pair_dist = tf.reshape(
            tf.tile(neg_pair_dist, [1, sent_im_ratio]), [num_sent, -1])
        sent_loss = tf.clip_by_value(
            self.margin + pos_pair_dist - neg_pair_dist, 0, 1e6)
        sent_loss = tf.reduce_mean(tf.nn.top_k(
            sent_loss, k=self.num_neg_sample)[0])

        # sentence only loss (neighborhood-preserving constraints)
        sent_sent_dist = pdist(sent_embeds, sent_embeds)
        sent_sent_mask = tf.reshape(tf.tile(tf.transpose(im_labels), [
                                    1, sent_im_ratio]), [num_sent, num_sent])
        pos_pair_dist = tf.reshape(tf.boolean_mask(
            sent_sent_dist, sent_sent_mask), [-1, sent_im_ratio])
        pos_pair_dist = tf.reduce_max(pos_pair_dist, axis=1, keep_dims=True)
        neg_pair_dist = tf.reshape(tf.boolean_mask(
            sent_sent_dist, ~sent_sent_mask), [num_sent, -1])
        sent_only_loss = tf.clip_by_value(
            self.margin + pos_pair_dist - neg_pair_dist, 0, 1e6)
        sent_only_loss = tf.reduce_mean(tf.nn.top_k(
            sent_only_loss, k=self.num_neg_sample)[0])

        loss = im_loss * self.im_loss_factor + sent_loss + \
            sent_only_loss * self.sent_only_loss_factor
        return loss


def embedding_loss_old(im_embeds, sent_embeds, im_labels, args):
    """
        im_embeds: (b, 512) image embedding tensors
        sent_embeds: (sample_size * b, 512) sentence embedding tensors
            where the order of sentence corresponds to the order of images and
            setnteces for the same image are next to each other
        im_labels: (sample_size * b, b) boolean tensor, where (i, j) entry is
            True if and only if sentence[i], image[j] is a positive pair
    """
    # compute embedding loss
    sent_im_ratio = args.sample_size
    num_img = args.batch_size
    num_sent = num_img * sent_im_ratio

    sent_im_dist = pdist(sent_embeds, im_embeds)
    # image loss: sentence, positive image, and negative image
    pos_pair_dist = tf.reshape(tf.boolean_mask(
        sent_im_dist, im_labels), [num_sent, 1])
    neg_pair_dist = tf.reshape(tf.boolean_mask(
        sent_im_dist, ~im_labels), [num_sent, -1])
    im_loss = tf.clip_by_value(
        args.margin + pos_pair_dist - neg_pair_dist, 0, 1e6)
    im_loss = tf.reduce_mean(tf.nn.top_k(im_loss, k=args.num_neg_sample)[0])
    # sentence loss: image, positive sentence, and negative sentence
    neg_pair_dist = tf.reshape(tf.boolean_mask(tf.transpose(
        sent_im_dist), ~tf.transpose(im_labels)), [num_img, -1])
    neg_pair_dist = tf.reshape(
        tf.tile(neg_pair_dist, [1, sent_im_ratio]), [num_sent, -1])
    sent_loss = tf.clip_by_value(
        args.margin + pos_pair_dist - neg_pair_dist, 0, 1e6)
    sent_loss = tf.reduce_mean(tf.nn.top_k(
        sent_loss, k=args.num_neg_sample)[0])
    # sentence only loss (neighborhood-preserving constraints)
    sent_sent_dist = pdist(sent_embeds, sent_embeds)
    sent_sent_mask = tf.reshape(tf.tile(tf.transpose(im_labels), [
                                1, sent_im_ratio]), [num_sent, num_sent])
    pos_pair_dist = tf.reshape(tf.boolean_mask(
        sent_sent_dist, sent_sent_mask), [-1, sent_im_ratio])
    pos_pair_dist = tf.reduce_max(pos_pair_dist, axis=1, keep_dims=True)
    neg_pair_dist = tf.reshape(tf.boolean_mask(
        sent_sent_dist, ~sent_sent_mask), [num_sent, -1])
    sent_only_loss = tf.clip_by_value(
        args.margin + pos_pair_dist - neg_pair_dist, 0, 1e6)
    sent_only_loss = tf.reduce_mean(tf.nn.top_k(
        sent_only_loss, k=args.num_neg_sample)[0])

    loss = im_loss * args.im_loss_factor + sent_loss + \
        sent_only_loss * args.sent_only_loss_factor
    return loss


def recall_k(im_embeds, sent_embeds, im_labels, ks=None):
    """
        Compute recall at given ks.
    """
    sent_im_dist = pdist(sent_embeds, im_embeds)

    def retrieval_recall(dist, labels, k):
        # Use negative distance to find the index of
        # the smallest k elements in each row.
        pred = tf.nn.top_k(-dist, k=k)[1]
        # Create a boolean mask for each column (k value) in pred,
        # s.t. mask[i][j] is 1 iff pred[i][k] = j.
        def pred_k_mask(topk_idx): return tf.one_hot(topk_idx, labels.shape[1],
                                                     on_value=True, off_value=False, dtype=tf.bool)
        # Create a boolean mask for the predicted indicies
        # by taking logical or of boolean masks for each column,
        # s.t. mask[i][j] is 1 iff j is in pred[i].
        pred_mask = tf.reduce_any(tf.map_fn(
            pred_k_mask, tf.transpose(pred), dtype=tf.bool), axis=0)
        # Entry (i, j) is matched iff pred_mask[i][j] and labels[i][j] are 1.
        matched = tf.cast(tf.logical_and(pred_mask, labels), dtype=tf.float32)
        return tf.reduce_mean(tf.reduce_max(matched, axis=1))
    return tf.concat(
        [tf.map_fn(lambda k: retrieval_recall(tf.transpose(sent_im_dist), tf.transpose(im_labels), k),
                   ks, dtype=tf.float32),
         tf.map_fn(lambda k: retrieval_recall(sent_im_dist, im_labels, k),
                   ks, dtype=tf.float32)],
        axis=0)


def setup_train_model(im_feats, sent_feats, train_phase, im_labels, args):
    # im_feats b x image_feature_dim
    # sent_feats 5b x sent_feature_dim
    # train_phase bool (Should be True.)
    # im_labels 5b x b
    i_embed, s_embed = embedding_model(
        im_feats, sent_feats, train_phase, im_labels)
    loss = embedding_loss(i_embed, s_embed, im_labels, args)
    return loss


def setup_eval_model(im_feats, sent_feats, train_phase, im_labels):
    # im_feats b x image_feature_dim
    # sent_feats 5b x sent_feature_dim
    # train_phase bool (Should be False.)
    # im_labels 5b x b
    i_embed, s_embed = embedding_model(
        im_feats, sent_feats, train_phase, im_labels)
    recall = recall_k(i_embed, s_embed, im_labels,
                      ks=tf.convert_to_tensor([1, 5, 10]))
    return recall


def setup_sent_eval_model(im_feats, sent_feats, train_phase, im_labels, args):
    # im_feats b x image_feature_dim
    # sent_feats 5b x sent_feature_dim
    # train_phase bool (Should be False.)
    # im_labels 5b x b
    _, s_embed = embedding_model(im_feats, sent_feats, train_phase, im_labels)
    # Create 5b x 5b sentence labels, wherthe 5 x 5 blocks along the diagonal
    num_sent = args.batch_size * args.sample_size
    sent_labels = tf.reshape(tf.tile(tf.transpose(im_labels),
                                     [1, args.sample_size]), [num_sent, num_sent])
    sent_labels = tf.logical_and(sent_labels, ~tf.eye(num_sent, dtype=tf.bool))
    # For topk, query k+1 since top1 is always the sentence itself, with dist 0.
    recall = recall_k(s_embed, s_embed, sent_labels,
                      ks=tf.convert_to_tensor([2, 6, 11]))[:3]
    return recall
