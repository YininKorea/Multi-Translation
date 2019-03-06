"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from torch.utils.serialization import load_lua
from torch.utils.data import DataLoader
from networks import Vgg16
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torchvision import transforms
from data import ImageFilelist, ImageFolder,MyDataSet
import torch
import os
import math
import torchvision.utils as vutils
import yaml
import numpy as np
import torch.nn.init as init
import time
import cv2
import torch.utils.data as dataf
# Methods
# get_all_data_loaders      : primary data loader interface (load trainA, testA, trainB, testB)
# get_data_loader_list      : list-based data loader
# get_data_loader_folder    : folder-based data loader
# get_config                : load yaml file
# eformat                   :
# write_2images             : save output image
# prepare_sub_folder        : create checkpoints and images folders for saving outputs
# write_one_row_html        : write one row of the html file for output images
# write_html                : create the html file.
# write_loss
# slerp
# get_slerp_interp
# get_model_list
# load_vgg16
# vgg_preprocess
# get_scheduler
# weights_init

def get_all_data_loaders(conf):
    batch_size = conf['batch_size']
    num_workers = conf['num_workers']
    if 'com_size' in conf:
        new_size_a = new_size_b = conf['com_size']
    else:
        new_size_a = conf['com_size_a']
        new_size_b = conf['com_size_b']

    if 'ins_size' in conf:
        ins_size = conf['ins_size']
    else:
        ins_size=128

    com_height = conf['com_image_height']
    com_width = conf['com_image_width']

    Ins_height = conf['ins_image_height']
    Ins_width = conf['ins_image_width']
    # identity_train_a = np.load('/root/UNIT/Data/train_a_identity.npy')
    # identity_train_b = np.load('/root/UNIT/Data/train_b_identity.npy')
    # identity_test_a = np.load('/root/UNIT/Data/test_a_identity.npy')
    # identity_test_b = np.load('/root/UNIT/Data/test_b_identity.npy')
    # train_Dataloader=torch.from_numpy(train_Dataloader)
    #
    # train_Dataloader = train_Dataloader.type(torch.FloatTensor)
    # #train_Dataloader=dataf.TensorDataset(train_Dataloader)
    # train_iden_a=torch.from_numpy(identity_train_a)
    # train_iden_a=train_iden_a.type(torch.FloatTensor)
    # train_iden_a=dataf.TensorDataset(train_iden_a)
    #
    # train_iden_b=torch.from_numpy(identity_train_b)
    # train_iden_b=train_iden_b.type(torch.FloatTensor)
    # train_iden_b=dataf.TensorDataset(train_iden_b)
    #
    # test_iden_a=torch.from_numpy(identity_test_a)
    # test_iden_a=test_iden_a.type(torch.FloatTensor)
    # test_iden_a=dataf.TensorDataset(test_iden_a)
    #
    # test_iden_b=torch.from_numpy(identity_test_b)
    # test_iden_b=test_iden_b.type(torch.FloatTensor)
    # test_iden_b=dataf.TensorDataset(test_iden_b)


    if 'data_root' in conf:
        train_loader_a = get_data_loader_folder(os.path.join(conf['data_root'], 'train_A'), batch_size, True,
                                              new_size_a, com_height, com_width, num_workers, True,mode=os.path.join(conf['data_root'],'train_a'))
        test_loader_a = get_data_loader_folder(os.path.join(conf['data_root'], 'test_A'), batch_size, False,
                                             new_size_a, new_size_a, new_size_a, num_workers, True,mode=os.path.join(conf['data_root'],'test_a'))
        train_loader_b = get_data_loader_folder(os.path.join(conf['data_root'], 'train_B'), batch_size, True,
                                              new_size_b,com_height, com_width, num_workers, True,mode=os.path.join(conf['data_root'],'train_b'))
        test_loader_b = get_data_loader_folder(os.path.join(conf['data_root'], 'test_B'), batch_size, False,
                                             new_size_b, new_size_b, new_size_b, num_workers, True,mode=os.path.join(conf['data_root'],'test_b'))
        train_loader_ins_a=get_data_loader_folder(os.path.join(conf['data_root'], 'train_ins_a'), batch_size, True,
                                              ins_size, Ins_height, Ins_width, num_workers, True,mode=None)
        test_loader_ins_a=get_data_loader_folder(os.path.join(conf['data_root'], 'test_ins_a'), batch_size, True,
                                              ins_size, Ins_height, Ins_width, num_workers, True,mode=None)

        train_loader_ins_b = get_data_loader_folder(os.path.join(conf['data_root'], 'train_ins_b'), batch_size, True,
                                                    ins_size, Ins_height, Ins_width, num_workers, True)
        test_loader_ins_b = get_data_loader_folder(os.path.join(conf['data_root'], 'test_ins_b'), batch_size, True,
                                                   ins_size, Ins_height, Ins_width, num_workers, True)



    else:
        train_loader_a = get_data_loader_list(conf['data_folder_train_a'], conf['data_list_train_a'], batch_size, True,
                                                new_size_a, com_height, com_width, num_workers, True)
        test_loader_a = get_data_loader_list(conf['data_folder_test_a'], conf['data_list_test_a'], batch_size, False,
                                                new_size_a, new_size_a, new_size_a, num_workers, True)
        train_loader_b = get_data_loader_list(conf['data_folder_train_b'], conf['data_list_train_b'], batch_size, True,
                                                new_size_b, com_height, com_width, num_workers, True)
        test_loader_b = get_data_loader_list(conf['data_folder_test_b'], conf['data_list_test_b'], batch_size, False,
                                                new_size_b, new_size_b, new_size_b, num_workers, True)
        train_loader_ins_a=get_data_loader_folder(os.path.join(conf['data_root'], 'train_ins_a'), batch_size, True,
                                              ins_size, Ins_height, Ins_width, num_workers, True)
        test_loader_ins_a=get_data_loader_folder(os.path.join(conf['data_root'], 'test_ins_a'), batch_size, True,
                                              ins_size, Ins_height, Ins_width, num_workers, True)

        train_loader_ins_b = get_data_loader_folder(os.path.join(conf['data_root'], 'train_ins_b'), batch_size, True,
                                                    ins_size, Ins_height, Ins_width, num_workers, True)
        test_loader_ins_b = get_data_loader_folder(os.path.join(conf['data_root'], 'test_ins_b'), batch_size, True,
                                                   ins_size, Ins_height, Ins_width, num_workers, True)
        #print("no")
    return train_loader_a, train_loader_b, test_loader_a, test_loader_b,train_loader_ins_a,test_loader_ins_a,train_loader_ins_b,test_loader_ins_b


def get_data_loader_list(root, file_list, batch_size, train, new_size=None,
                           height=256, width=256, num_workers=4, crop=True,mode='train_a'):
    transform_list = [transforms.ToTensor(),
                      transforms.Normalize((0.5, 0.5, 0.5),
                                           (0.5, 0.5, 0.5))]
    #transform_list = [transforms.RandomCrop((height, width))] + transform_list if crop else transform_list
    transform_list = [transforms.Resize(new_size)] + transform_list if new_size is not None else transform_list
    #transform_list = [transforms.RandomHorizontalFlip()] + transform_list if train else transform_list
    transform = transforms.Compose(transform_list)
    dataset = ImageFilelist(file_list, transform=transform)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=train, drop_last=True, num_workers=num_workers)
    return loader

# def get_data_loader_folder(input_folder, batch_size, train, new_size=None,
#                            height=256, width=256, num_workers=4, crop=True,mode=None):
#     transform_list = [transforms.ToTensor(),
#                       transforms.Normalize((0.5, 0.5, 0.5),
#                                            (0.5, 0.5, 0.5))]
#     transform_list = [transforms.RandomCrop((height, width))] + transform_list if crop else transform_list
#     transform_list = [transforms.Resize(new_size)] + transform_list if new_size is not None else transform_list
#     transform_list = [transforms.RandomHorizontalFlip()] + transform_list if train else transform_list
#     transform = transforms.Compose(transform_list)
#     dataset= ImageFolder(input_folder, transform=transform,return_paths=True)
#     dataset_2= ImageFolder(input_folder, transform=transform)
#     #print(len(dataset[0]))
#     split_path(dataset[0][1],mode=mode)
#     #test_pad()
#     loader = DataLoader(dataset=dataset_2, batch_size=batch_size, shuffle=train, drop_last=True, num_workers=num_workers)
#     return loader
def get_data_loader_folder(input_folder, batch_size, train, new_size=None,height=256, width=256, num_workers=4, crop=True,mode=None):
    transform_list = [transforms.ToTensor(),
                      transforms.Normalize((0.5, 0.5, 0.5),
                                           (0.5, 0.5, 0.5))]
    #transform_list = [transforms.RandomCrop((height, width))] + transform_list if crop else transform_list
    #transform_list = [transforms.Resize(new_size)] + transform_list if new_size is not None else transform_list
    #transform_list = [transforms.RandomHorizontalFlip()] + transform_list if train else transform_list

    transform = transforms.Compose(transform_list)

    dataset= MyDataSet(input_folder, transform=transform,return_paths=True)
    dataset_2=MyDataSet(input_folder, transform=transform)
    split_path(dataset[0][1],mode=mode)
    loader = DataLoader(dataset=dataset_2, batch_size=batch_size, shuffle=train, drop_last=True, num_workers=num_workers)
    return loader



def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream)

def split_path(img_path,mode=None):
    if mode is not None:
        #print(mode)
        id_npy=np.load(mode+'_identity.npy')
        #print(id_npy.shape)
        new_npy=np.zeros(id_npy.shape)
        i=0
        for path in img_path:
            #print(path)
            id=str(path).split('/')[3].split('.')[0]
            #print(id)
            new_npy[i]=id_npy[int(id)%150]
            #print(id_npy[int(id)%150])
            i+=1

        np.save(mode+'_identity_2.npy',new_npy)
def test_pad():
    id_npy = np.load('/root/UNIT/Data/train_a_identity_2.npy')
    id=np.load('/root/UNIT/Data/train_a_id_2.npy')
    #print(id.shape)
    for i in range(150):
        #print(str(int(id[i])))
        background = cv2.imread('/root/UNIT/Data/train_A/'+str(int(id[i]))+'.jpg')
        print(background.shape)
        ymin = int(id_npy[i][1] - 1)
        ymax = int(id_npy[i][3] - 1)
        xmin = int(id_npy[i][0] - 1)
        xmax = int(id_npy[i][2] - 1)
        background = cv2.resize(background, (256, 256))
        shape = background[ymin:ymax, xmin:xmax, :].shape
        ins_a = cv2.imread('/root/UNIT/Data/train_ins_a/' + str(int(id[i])) + '.jpg')
        ins_a = cv2.resize(ins_a, (shape[1], shape[0]))
        background[ymin:ymax, xmin:xmax, :] = ins_a
        cv2.imwrite('/root/UNIT/Data/test_pad/' + str(i) + '.jpg', background)

def eformat(f, prec):
    s = "%.*e"%(prec, f)
    mantissa, exp = s.split('e')
    # add 1 to digits as 1 is taken by sign +/-
    return "%se%d"%(mantissa, int(exp))


def __write_images(image_outputs, display_image_num, file_name):
    image_outputs = [images.expand(-1, 3, -1, -1) for images in image_outputs] # expand gray-scale images to 3 channels
    image_tensor = torch.cat([images[:display_image_num] for images in image_outputs], 0)
    image_grid = vutils.make_grid(image_tensor.data, nrow=display_image_num, padding=0, normalize=True)
    vutils.save_image(image_grid, file_name, nrow=1)
    #print(file_name)
def __write_images_2(image_outputs, display_image_num, file_name):
    #image_outputs = [images.expand(-1, 3, -1, -1) for images in image_outputs] # expand gray-scale images to 3 channels
    #for i in range(display_image_num):
    image_tensor = torch.cat([images[:display_image_num] for images in image_outputs], 1)
    #print(type(image_tensor))
    image_grid = vutils.make_grid(image_tensor.data)
    #print(type(image_grid))
    vutils.save_image(image_grid, file_name, nrow=1)

def MUNIT_write_2images(image_outputs, display_image_num, image_directory, postfix,index,mode='train'):
    #n = len(image_outputs)
    #print(n)
    #single=image_outputs[0]
    #print(type(single))
    #print(image_outputs[0:(n-4)//2].shape)
    __write_images(image_outputs[0:4], display_image_num, '%s/gen_a_recon_trans_%s.jpg' % (image_directory, postfix))
    __write_images(image_outputs[4:8], display_image_num, '%s/gen_b_recon_trans_%s.jpg' % (image_directory, postfix))
    __write_images(image_outputs[8:12],  display_image_num,'%s/gen_ins_a_recon_trans_%s.jpg' % (image_directory, postfix))
    __write_images(image_outputs[12:16], display_image_num, '%s/gen_ins_b_recon_trans_%s.jpg' % (image_directory, postfix))
    __write_images(image_outputs[16:21],display_image_num,'%s/padding_results_ab_%s.jpg' % (image_directory, postfix))
    __write_images(image_outputs[21:26], display_image_num, '%s/padding_results_ba_%s.jpg' % (image_directory, postfix))
# def write_2images(image_outputs, display_image_num, image_directory, postfix,iden_a,iden_b,mode='train'):
#     n = len(image_outputs)
#     #single=image_outputs[0]
#     #print(type(single))
#     #print(image_outputs[0:(n-4)//2].shape)
#     __write_images(image_outputs[0:(n-4)//2], display_image_num, '%s/gen_a2b_%s.jpg' % (image_directory, postfix))
#     __write_images(image_outputs[(n-4)//2:n-4], display_image_num, '%s/gen_b2a_%s.jpg' % (image_directory, postfix))
#     __write_images(image_outputs[(n-4):n-2],display_image_num,'%s/gen_ins_a_%s.jpg' % (image_directory, postfix))
#     __write_images(image_outputs[(n - 2):n], display_image_num, '%s/gen_ins_b_%s.jpg' % (image_directory, postfix))
#     result_pad(image_outputs[6],image_outputs[0],postfix,iden_a,mode=mode,ins='a',name='ori_a')
#     result_pad(image_outputs[8], image_outputs[3],postfix, iden_b,mode=mode,ins='b',name='ori_b')
#     result_pad(image_outputs[7], image_outputs[1],postfix, iden_a,mode=mode,ins='a',name='re_a')
#     result_pad(image_outputs[9], image_outputs[4],postfix, iden_b,mode=mode,ins='b',name='re_b')
#     result_pad(image_outputs[7],image_outputs[5],postfix,iden_a,mode=mode,ins='b',name='tran_a')
#     result_pad(image_outputs[9], image_outputs[2],postfix, iden_b,mode=mode,ins='a',name='tran_b')

def write_2images(image_outputs, display_image_num, image_directory, postfix,index,mode='train'):
    #n = len(image_outputs)
    #print(n)
    #single=image_outputs[0]
    #print(type(single))
    #print(image_outputs[0:(n-4)//2].shape)
    __write_images(image_outputs[0:3], display_image_num, '%s/gen_a2b_%s.jpg' % (image_directory, postfix))
    __write_images(image_outputs[3:6], display_image_num, '%s/gen_b2a_%s.jpg' % (image_directory, postfix))
    __write_images(image_outputs[6:8],display_image_num,'%s/gen_ins_a_%s.jpg' % (image_directory, postfix))
    __write_images(image_outputs[8:10], display_image_num, '%s/gen_ins_b_%s.jpg' % (image_directory, postfix))
    __write_images(image_outputs[10:16],display_image_num,'%s/padding_results_%s.jpg' % (image_directory, postfix))



    # ori_a=result_pad(index,image_outputs[6],image_outputs[0],postfix,mode=mode,ins='a',name='ori_a')
    # ori_b=result_pad(index, image_outputs[8], image_outputs[3],postfix, mode=mode,ins='b',name='ori_b')
    # recon_a=result_pad(index,image_outputs[7], image_outputs[1],postfix, mode=mode,ins='a',name='re_a')
    # recon_b=result_pad(index, image_outputs[9], image_outputs[4],postfix, mode=mode,ins='b',name='re_b')
    # trans_a=result_pad(index,image_outputs[7],image_outputs[5],postfix,mode=mode,ins='b',name='tran_a')
    # trans_b=result_pad(index, image_outputs[9], image_outputs[2],postfix, mode=mode,ins='a',name='tran_b')
    # original_a = result_pad(index,image_outputs[6],image_outputs[0],postfix,mode=mode,ins='a')
    # original_b = result_pad(index, image_outputs[6], image_outputs[0],postfix, mode=mode,ins='b')
    # recontructed_a=result_pad(index,image_outputs[7], image_outputs[1],postfix, mode=mode,ins='a')
    # recontructed_b=result_pad(index, image_outputs[8], image_outputs[4],postfix, mode=mode,ins='b')
    # trainslatedb=result_pad(index,image_outputs[9],image_outputs[2],postfix,mode=mode,ins='b')
    # # trainslateda=result_pad(index, image_outputs[7], image_outputs[5],postfix, mode=mode,ins='a')
    # original_path="./Data/result_outputs/"+mode+''
    #
    # isExist=os.path.exists(original_path)
    # if not isExist:
    #     os.mkdir(original_path)
    # __write_images_2(ori_a, display_image_num, '%s/original_a_%s.jpg' % (original_path,postfix))
    # __write_images_2(ori_b, display_image_num, '%s/original_b_%s.jpg' % (original_path, postfix))
    # __write_images_2(recon_a, display_image_num, '%s/recon_a_%s.jpg' % (original_path,postfix))
    # __write_images_2(recon_b, display_image_num, '%s/recon_b_%s.jpg' % (original_path,postfix))
    # __write_images_2(trans_a, display_image_num, '%s/trans_a_%s.jpg' % (original_path,postfix))
    # __write_images_2(trans_b, display_image_num, '%s/trans_b_%s.jpg' % (original_path,postfix))


def prepare_sub_folder(output_directory):
    image_directory = os.path.join(output_directory, 'images')
    if not os.path.exists(image_directory):
        print("Creating directory: {}".format(image_directory))
        os.makedirs(image_directory)
    checkpoint_directory = os.path.join(output_directory, 'checkpoints')
    if not os.path.exists(checkpoint_directory):
        print("Creating directory: {}".format(checkpoint_directory))
        os.makedirs(checkpoint_directory)
    return checkpoint_directory, image_directory


def write_one_row_html(html_file, iterations, img_filename, all_size):
    html_file.write("<h3>iteration [%d] (%s)</h3>" % (iterations,img_filename.split('/')[-1]))
    html_file.write("""
        <p><a href="%s">
          <img src="%s" style="width:%dpx">
        </a><br>
        <p>
        """ % (img_filename, img_filename, all_size))
    return


def write_html(filename, iterations, image_save_iterations, image_directory, all_size=1536):
    html_file = open(filename, "w")
    html_file.write('''
    <!DOCTYPE html>
    <html>
    <head>
      <title>Experiment name = %s</title>
      <meta http-equiv="refresh" content="30">
    </head>
    <body>
    ''' % os.path.basename(filename))
    html_file.write("<h3>current</h3>")
    write_one_row_html(html_file, iterations, '%s/gen_a2b_train_current.jpg' % (image_directory), all_size)
    write_one_row_html(html_file, iterations, '%s/gen_b2a_train_current.jpg' % (image_directory), all_size)
    for j in range(iterations, image_save_iterations-1, -1):
        if j % image_save_iterations == 0:
            write_one_row_html(html_file, j, '%s/gen_a2b_test_%08d.jpg' % (image_directory, j), all_size)
            write_one_row_html(html_file, j, '%s/gen_b2a_test_%08d.jpg' % (image_directory, j), all_size)
            write_one_row_html(html_file, j, '%s/gen_a2b_train_%08d.jpg' % (image_directory, j), all_size)
            write_one_row_html(html_file, j, '%s/gen_b2a_train_%08d.jpg' % (image_directory, j), all_size)
    html_file.write("</body></html>")
    html_file.close()


def write_loss(iterations, trainer, train_writer):
    members = [attr for attr in dir(trainer) \
               if not callable(getattr(trainer, attr)) and not attr.startswith("__") and ('loss' in attr or 'grad' in attr or 'nwd' in attr)]
    for m in members:
        train_writer.add_scalar(m, getattr(trainer, m), iterations + 1)


def slerp(val, low, high):
    """
    original: Animating Rotation with Quaternion Curves, Ken Shoemake
    https://arxiv.org/abs/1609.04468
    Code: https://github.com/soumith/dcgan.torch/issues/14, Tom White
    """
    omega = np.arccos(np.dot(low / np.linalg.norm(low), high / np.linalg.norm(high)))
    so = np.sin(omega)
    return np.sin((1.0 - val) * omega) / so * low + np.sin(val * omega) / so * high


def get_slerp_interp(nb_latents, nb_interp, z_dim):
    """
    modified from: PyTorch inference for "Progressive Growing of GANs" with CelebA snapshot
    https://github.com/ptrblck/prog_gans_pytorch_inference
    """

    latent_interps = np.empty(shape=(0, z_dim), dtype=np.float32)
    for _ in range(nb_latents):
        low = np.random.randn(z_dim)
        high = np.random.randn(z_dim)  # low + np.random.randn(512) * 0.7
        interp_vals = np.linspace(0, 1, num=nb_interp)
        latent_interp = np.array([slerp(v, low, high) for v in interp_vals],
                                 dtype=np.float32)
        latent_interps = np.vstack((latent_interps, latent_interp))

    return latent_interps[:, :, np.newaxis, np.newaxis]


# Get model list for resume
def get_model_list(dirname, key):
    if os.path.exists(dirname) is False:
        return None
    gen_models = [os.path.join(dirname, f) for f in os.listdir(dirname) if
                  os.path.isfile(os.path.join(dirname, f)) and key in f and ".pt" in f]
    if gen_models is None:
        return None
    gen_models.sort()
    last_model_name = gen_models[-1]
    return last_model_name


def load_vgg16(model_dir):
    """ Use the model from https://github.com/abhiskk/fast-neural-style/blob/master/neural_style/utils.py """
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    if not os.path.exists(os.path.join(model_dir, 'vgg16.weight')):
        if not os.path.exists(os.path.join(model_dir, 'vgg16.t7')):
            os.system('wget https://www.dropbox.com/s/76l3rt4kyi3s8x7/vgg16.t7?dl=1 -O ' + os.path.join(model_dir, 'vgg16.t7'))
        vgglua = load_lua(os.path.join(model_dir, 'vgg16.t7'))
        vgg = Vgg16()
        for (src, dst) in zip(vgglua.parameters()[0], vgg.parameters()):
            dst.data[:] = src
        torch.save(vgg.state_dict(), os.path.join(model_dir, 'vgg16.weight'))
    vgg = Vgg16()
    vgg.load_state_dict(torch.load(os.path.join(model_dir, 'vgg16.weight')))
    return vgg

# def result_pad(instance,background,profix,iden,mode="train",ins="a",name="ori"):
#
#
#     background=background.cpu().numpy()
#     instance=instance.cpu().numpy()
#     original_path="./Data/result_outputs/"+mode+''
#
#     isExist=os.path.exists(original_path)
#     if not isExist:
#         os.mkdir(original_path)
#     #iden=iden[0:5]
#     #print(type(iden))
#     #print(type(iden[0]))
#     #iden = np.array(iden, dtype=float)
#     for tensor in iden:
#
#         ymin = int(tensor[1] - 1)
#         ymax = int(tensor[3] - 1)
#         xmin = int(tensor[0] - 1)
#         xmax = int(tensor[2] - 1)
#         padding=np.transpose(background[0],(1,2,0))
#         padding = cv2.resize(padding,(512,512))
#         #print(padding.shape)
#         shape = padding[ymin:ymax, xmin:xmax,:].shape
#         instance_a=np.transpose(instance[0],(1,2,0))
#         ins_a = cv2.resize(instance_a, (shape[1], shape[0]))
#         padding[ymin:ymax, xmin:xmax,:] = ins_a
#         padding=torch.from_numpy(np.transpose(padding,(2,0,1)))
#         path=original_path+'/'+str(name)+'_'+profix+'.jpg'
#         image_grid = vutils.make_grid(padding.data)
#         vutils.save_image(image_grid, path, nrow=1)
def result_pad(index,instance,background,profix,mode="train",ins="a",name="ori"):
    save_name="./Data/"+str(mode)+"_"+str(ins)+"_identity.npy"
    #print(save_name)
    identity=np.load(save_name)
    #print(identity.shape)
    background=background.cpu().numpy()
    instance=instance.cpu().numpy()
    original_path="./Data/result_outputs/"+mode+''
    visu=[]
    isExist=os.path.exists(original_path)
    if not isExist:
        os.mkdir(original_path)
    for i in range(10):
        ymin = int(identity[i][1] - 1)
        ymax = int(identity[i][3] - 1)
        xmin = int(identity[i][0] - 1)
        xmax = int(identity[i][2] - 1)
        padding=np.transpose(background[i],(1,2,0))
        padding = cv2.resize(padding,(512,512))
        shape = padding[ymin:ymax, xmin:xmax,:].shape
        instance_a=np.transpose(instance[i],(1,2,0))
        ins_a = cv2.resize(instance_a, (shape[1], shape[0]))
        padding[ymin:ymax, xmin:xmax,:] = ins_a
        padding=torch.from_numpy(np.transpose(padding,(2,0,1)))
        # path=original_path+'/'+str(name)+'_'+str(ins)+'_'+profix+'_'+str(i)+'.jpg'
        # image_grid = vutils.make_grid(padding.data)
        # vutils.save_image(image_grid, path, nrow=1)
        visu.append(padding)
    return visu



def vgg_preprocess(batch):
    tensortype = type(batch.data)
    (r, g, b) = torch.chunk(batch, 3, dim = 1)
    batch = torch.cat((b, g, r), dim = 1) # convert RGB to BGR
    batch = (batch + 1) * 255 * 0.5 # [-1, 1] -> [0, 255]
    mean = tensortype(batch.data.size()).cuda()
    mean[:, 0, :, :] = 103.939
    mean[:, 1, :, :] = 116.779
    mean[:, 2, :, :] = 123.680
    batch = batch.sub(Variable(mean)) # subtract mean
    return batch


def get_scheduler(optimizer, hyperparameters, iterations=-1):
    if 'lr_policy' not in hyperparameters or hyperparameters['lr_policy'] == 'constant':
        scheduler = None # constant scheduler
    elif hyperparameters['lr_policy'] == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=hyperparameters['step_size'],
                                        gamma=hyperparameters['gamma'], last_epoch=iterations)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', hyperparameters['lr_policy'])
    return scheduler


def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
            # print m.__class__.__name__
            if init_type == 'gaussian':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

    return init_fun


class Timer:
    def __init__(self, msg):
        self.msg = msg
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_value, exc_tb):
        print(self.msg % (time.time() - self.start_time))


def pytorch03_to_pytorch04(state_dict_base):
    def __conversion_core(state_dict_base):
        state_dict = state_dict_base.copy()
        for key, value in state_dict_base.items():
            if key.endswith(('enc.model.0.norm.running_mean',
                             'enc.model.0.norm.running_var',
                             'enc.model.1.norm.running_mean',
                             'enc.model.1.norm.running_var',
                             'enc.model.2.norm.running_mean',
                             'enc.model.2.norm.running_var',
                             'enc.model.3.model.0.model.1.norm.running_mean',
                             'enc.model.3.model.0.model.1.norm.running_var',
                             'enc.model.3.model.0.model.0.norm.running_mean',
                             'enc.model.3.model.0.model.0.norm.running_var',
                             'enc.model.3.model.1.model.1.norm.running_mean',
                             'enc.model.3.model.1.model.1.norm.running_var',
                             'enc.model.3.model.1.model.0.norm.running_mean',
                             'enc.model.3.model.1.model.0.norm.running_var',
                             'enc.model.3.model.2.model.1.norm.running_mean',
                             'enc.model.3.model.2.model.1.norm.running_var',
                             'enc.model.3.model.2.model.0.norm.running_mean',
                             'enc.model.3.model.2.model.0.norm.running_var',
                             'enc.model.3.model.3.model.1.norm.running_mean',
                             'enc.model.3.model.3.model.1.norm.running_var',
                             'enc.model.3.model.3.model.0.norm.running_mean',
                             'enc.model.3.model.3.model.0.norm.running_var',
                             'dec.model.0.model.0.model.1.norm.running_mean',
                             'dec.model.0.model.0.model.1.norm.running_var',
                             'dec.model.0.model.0.model.0.norm.running_mean',
                             'dec.model.0.model.0.model.0.norm.running_var',
                             'dec.model.0.model.1.model.1.norm.running_mean',
                             'dec.model.0.model.1.model.1.norm.running_var',
                             'dec.model.0.model.1.model.0.norm.running_mean',
                             'dec.model.0.model.1.model.0.norm.running_var',
                             'dec.model.0.model.2.model.1.norm.running_mean',
                             'dec.model.0.model.2.model.1.norm.running_var',
                             'dec.model.0.model.2.model.0.norm.running_mean',
                             'dec.model.0.model.2.model.0.norm.running_var',
                             'dec.model.0.model.3.model.1.norm.running_mean',
                             'dec.model.0.model.3.model.1.norm.running_var',
                             'dec.model.0.model.3.model.0.norm.running_mean',
                             'dec.model.0.model.3.model.0.norm.running_var',
                             )):
                del state_dict[key]
        return state_dict
    state_dict = dict()
    state_dict['a'] = __conversion_core(state_dict_base['a'])
    state_dict['b'] = __conversion_core(state_dict_base['b'])
    return state_dict