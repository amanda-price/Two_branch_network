import os.path as p

from .dataset_utils import DatasetLoader
from .retrieval_model import EmbeddingLoss, setup_train_model, embedding_model

def train_step():
    pass

def main(config: dict):
    image_feat_path = p.abspath(config['image_feat_path'])
    sent_feat_path = p.abspath(config['sent_feat_path'])

    # Load data.
    data_loader = DatasetLoader(image_feat_path,sent_feat_path)
    #num_ims, im_feat_dim = data_loader.im_feat_shape
    #num_sents, sent_feat_dim = data_loader.sent_feat_shape
    #steps_per_epoch = num_sents // args['batch_size']
    #num_steps = steps_per_epoch * args['max_num_epoch']

    model = embedding_model(data_loader.im_feats.shape, data_loader.sent_feats.shape)
    loss = EmbeddingLoss(**config)

    img_feats = data_loader.im_feats
    sent_feats = data_loader.sent_feats

    # for epoch in config["epochs"]:


    # Setup training operation.
    # loss = setup_train_model(data_loader.im_feats, data_loader.sent_feats)
    #print(summary(loss))
    # Setup optimizer.

    # Setup model saver.



