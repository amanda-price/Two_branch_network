

from .dataset_utils import DatasetLoader
from .retrieval_model import setup_train_model

def main(args):
    image_feat_path = args['image_feat_path']
    sent_feat_path = args['sent_feat_path']

    # Load data.
    data_loader = DatasetLoader(image_feat_path,sent_feat_path)
    #num_ims, im_feat_dim = data_loader.im_feat_shape
    #num_sents, sent_feat_dim = data_loader.sent_feat_shape
    #steps_per_epoch = num_sents // args['batch_size']
    #num_steps = steps_per_epoch * args['max_num_epoch']

    # Setup training operation.
    loss = setup_train_model(data_loader.im_feats, data_loader.sent_feats) 
    #print(summary(loss))
    # Setup optimizer.
    
    # Setup model saver.
    


