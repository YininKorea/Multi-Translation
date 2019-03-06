"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from utils import get_all_data_loaders, prepare_sub_folder, write_html, write_loss, get_config, write_2images, Timer,MUNIT_write_2images
import argparse
from torch.autograd import Variable
from trainer import MUNIT_Trainer, UNIT_Trainer
import torch.backends.cudnn as cudnn
import torch
from GetConfigOptions import VOCDataloader
try:
    from itertools import izip as zip
except ImportError: # will be 3.x series
    pass
import os
import sys
import tensorboardX
import shutil
import numpy as np

parser = argparse.ArgumentParser(description="config")
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/edges2handbags_folder.yaml', help='Path to the config file.')
parser.add_argument('--output_path', type=str, default='.', help="outputs path")
parser.add_argument("--resume", action="store_true")
parser.add_argument('--trainer', type=str, default='MUNIT', help="MUNIT|UNIT")
opts = parser.parse_args()
classname = ['cat', 'dog']
cudnn.benchmark = True

# Load experiment setting
config = get_config(opts.config)
max_iter = config['max_iter']
display_size = config['display_size']
config['vgg_model_path'] = opts.output_path
pascal = VOCDataloader(config, classname)
# #input_A,input_instance,inputB=pascal.train_loader(classname)
pascal.train_loader(classname)

print("dataset has been prepared!!")
# i=0
# Setup model and data loader
if opts.trainer == 'MUNIT':
    trainer = MUNIT_Trainer(config)
elif opts.trainer == 'UNIT':
    trainer = UNIT_Trainer(config)
else:
    sys.exit("Only support MUNIT|UNIT")
trainer.cuda()
train_loader_a, train_loader_b, test_loader_a, test_loader_b, train_loader_ins_a,test_loader_ins_a,train_loader_ins_b,test_loader_ins_b= get_all_data_loaders(config)

train_display_images_a = torch.stack([train_loader_a.dataset[i] for i in range(display_size)]).cuda()
train_display_images_b = torch.stack([train_loader_b.dataset[i] for i in range(display_size)]).cuda()
train_display_images_ins_a=torch.stack([train_loader_ins_a.dataset[i] for i in range(display_size)]).cuda()
train_display_images_ins_b=torch.stack([train_loader_ins_b.dataset[i] for i in range(display_size)]).cuda()


save_name = "./Data/train_a_smooth.npy"
smooth_a = np.load(save_name)
#smooth_a=torch.from_numpy(smooth_a).float().cuda()
# print(save_name)
print(smooth_a.shape)
save_name = "./Data/train_b_smooth.npy"
smooth_b = np.load(save_name)
print(smooth_b.shape)
#smooth_b=torch.from_numpy(smooth_b).float().cuda()
# print(train_iden_a[0])
# Mat_train_a=[]
# Mat_train_b=[]
# Mat_test_a=[]
# Mat_test_b=[]
# print(type(train_iden_a[0]))
# for i in range(display_size):
#     Mat_train_a.append(train_iden_a[i])
#     Mat_train_b.append(train_iden_b[i])
#     Mat_test_a.append(test_iden_a[i])
#     Mat_test_b.append(test_iden_b[i])

# train_dis_id_a=torch.cat([train_iden_a[i]] for i in range(display_size)).cuda()
# train_dis_id_b=torch.cat([train_iden_b[i]] for i in range(display_size)).cuda()

test_display_images_a = torch.stack([test_loader_a.dataset[i] for i in range(display_size)]).cuda()
test_display_images_b = torch.stack([test_loader_b.dataset[i] for i in range(display_size)]).cuda()
test_display_images_ins_a = torch.stack([test_loader_ins_a.dataset[i] for i in range(display_size)]).cuda()
test_display_images_ins_b= torch.stack([test_loader_ins_b.dataset[i] for i in range(display_size)]).cuda()
# test_dis_id_a=torch.cat([test_iden_a[i]] for i in range(display_size)).cuda()
# test_dis_id_b=torch.cat([test_iden_b[i]] for i in range(display_size)).cuda()
save_name = "./Data/train_a_identity_2.npy"
identity_a = np.load(save_name)
# print(save_name)
save_name = "./Data/train_b_identity_2.npy"
identity_b = np.load(save_name)
# print(save_name)
original_path = "./Data/result_outputs/train/"
isExist = os.path.exists(original_path)
if not isExist:
    os.mkdir(original_path)
original_path = "./Data/result_outputs/test/"
isExist = os.path.exists(original_path)
if not isExist:
    os.mkdir(original_path)
# Setup logger and output folders
model_name = os.path.splitext(os.path.basename(opts.config))[0]
train_writer = tensorboardX.SummaryWriter(os.path.join(opts.output_path + "/logs", model_name))
output_directory = os.path.join(opts.output_path + "/outputs", model_name)
checkpoint_directory, image_directory = prepare_sub_folder(output_directory)
shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml')) # copy config file to output folder


# Start training
iterations = trainer.resume(checkpoint_directory, hyperparameters=config) if opts.resume else 0
while True:
    index=0
    for it, (images_a, images_b,images_ins_a,images_ins_b) in enumerate(zip(train_loader_a, train_loader_b,train_loader_ins_a,train_loader_ins_b)):
        trainer.update_learning_rate(config['end-to-end'])
        images_a, images_b,images_ins_a,images_ins_b = images_a.cuda().detach(), images_b.cuda().detach(),images_ins_a.cuda().detach(),images_ins_b.cuda().detach()
        if config['end-to-end']:
            #print("hello")
            with Timer("end-to-end model elapsed time in update: %f"):
                # M
                trainer.End_mode_Dis_update(images_a, images_ins_a, images_b, images_ins_b,index,config,smooth_a[index],smooth_b[index])
                trainer.End_mode_Gen_update(images_a, images_ins_a, images_b, images_ins_b,index,config)
                torch.cuda.synchronize()

        else:
            with Timer("Elapsed time in update: %f"):
                # M
                trainer.Dis_update(images_a,images_ins_a, images_b,images_ins_b, config)
                trainer.Gen_update(images_a,images_ins_a,images_b,images_ins_b, config)
                torch.cuda.synchronize()
        #print("hello")
        # Dump training stats in log file
        if (iterations + 1) % config['log_iter'] == 0:
            print("Iteration: %08d/%08d" % (iterations + 1, max_iter))
            write_loss(iterations, trainer, train_writer)

        # Write images
        if (iterations + 1) % config['image_save_iter'] == 0:
            with torch.no_grad():
                test_image_outputs = trainer.sample(test_display_images_a, test_display_images_b,test_display_images_ins_a,test_display_images_ins_b,mode='test')
                #print(type(test_image_outputs))
                train_image_outputs = trainer.sample(train_display_images_a, train_display_images_b,train_display_images_ins_a,train_display_images_ins_b,mode='train')
            # write_2images(test_image_outputs, display_size, image_directory, 'test_%08d' % (iterations + 1),index,mode='test')
            # write_2images(train_image_outputs, display_size, image_directory, 'train_%08d' % (iterations + 1),index,mode='train')
            if opts.trainer == 'UNIT':
                write_2images(test_image_outputs, display_size, image_directory, 'test_%08d' % (iterations + 1),index,mode='test')
                write_2images(train_image_outputs, display_size, image_directory, 'train_%08d' % (iterations + 1),index,mode='train')

                # HTML
                write_html(output_directory + "/index.html", iterations + 1, config['image_save_iter'], 'images')

                if (iterations + 1) % config['image_display_iter'] == 0:
                    with torch.no_grad():
                        image_outputs = trainer.sample(train_display_images_a, train_display_images_b,train_display_images_ins_a,train_display_images_ins_b,mode='train')
                    write_2images(image_outputs, display_size, image_directory, 'train_current',index,mode='train')

                # Save network weights
                if (iterations + 1) % config['snapshot_save_iter'] == 0:
                    trainer.save(checkpoint_directory, iterations,config['end-to-end'])

                iterations += 1
                index+=1
                if iterations >= max_iter:
                    sys.exit('Finish training')

            if opts.trainer == 'MUNIT':
                #print("hello")
                MUNIT_write_2images(test_image_outputs, display_size, image_directory, 'test_%08d' % (iterations + 1),index,mode='test')
                MUNIT_write_2images(train_image_outputs, display_size, image_directory, 'train_%08d' % (iterations + 1),index,mode='train')

                # HTML
                write_html(output_directory + "/index.html", iterations + 1, config['image_save_iter'], 'images')

                if (iterations + 1) % config['image_display_iter'] == 0:
                    with torch.no_grad():
                        image_outputs = trainer.sample(train_display_images_a, train_display_images_b,train_display_images_ins_a,train_display_images_ins_b,mode='train')
                    write_2images(image_outputs, display_size, image_directory, 'train_current',index,mode='train')

                # Save network weights
                if (iterations + 1) % config['snapshot_save_iter'] == 0:
                    trainer.save(checkpoint_directory, iterations,config['end-to-end'])

        iterations += 1
        index+=1
        if iterations >= max_iter:
            sys.exit('Finish training')


