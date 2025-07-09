# -*- coding: utf-8 -*-
from tqdm.notebook import tqdm
import numpy as np
import torch
import os
import datetime

def save_model(model, folder_path, name):
  timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
  PATH = os.path.join(folder_path, name  + '_{}'.format(timestamp))
  torch.save(model.state_dict(), PATH)
  
def train(model, criterion, optimizer, sheduler, dataloader, device, epochs, 
          generator_folder_path, discriminator_folder_path):

  model["discriminator"].train()
  model["generator"].train()
  torch.cuda.empty_cache()

  #Losses
  losses_g = []
  losses_d = []

  best_loss_g = float('inf')
  best_loss_d = float('inf')
  for epoch in tqdm(range(epochs), desc= str(epochs)  + " Epochs"):

    loss_g_per_epoch = []
    loss_d_per_epoch = []


    for Image_HR, Image_LR in tqdm(dataloader, desc= "epoch{" + str(epoch+1) + "}: "  ):
      batch_size = Image_HR.size(0)

      #Training discriminator
      optimizer["discriminator"].zero_grad()
      GImage_HR = model["generator"](Image_LR) #Generate images
      pred = model["discriminator"](Image_HR)
      target = torch.ones(batch_size, 1, device=device)
      loss1 = criterion["discriminator"](pred, target)

      pred = model["discriminator"](GImage_HR)
      target = torch.zeros(batch_size, 1, device=device)
      loss2 = criterion["discriminator"](pred, target)

      loss_d = loss1 + loss2
      loss_d.backward()
      optimizer["discriminator"].step()
      sheduler["discriminator"].step()
      loss_d_per_epoch.append(loss_d.item())

      # Train generator
      optimizer["generator"].zero_grad()

      # Try to fool the discriminator
      GImage_HR = model["generator"](Image_LR) #Generate images
      pred = model["discriminator"](GImage_HR)
      target = torch.ones(batch_size, 1, device=device)
      loss_g = criterion["generator"](GImage_HR, Image_HR, pred, target)

      loss_g.backward()
      optimizer["generator"].step()
      sheduler["generator"].step()
      loss_g_per_epoch.append(loss_g.item())

    # Record losses & scores
    losses_g.append(np.mean(loss_g_per_epoch))
    losses_d.append(np.mean(loss_d_per_epoch))

    print("G --> last_loss: ", losses_g[-1], " || best_loss: ", best_loss_g )
    print("D --> last_loss: ", losses_d[-1], " || best_loss: ", best_loss_d )
    if(best_loss_g > losses_g[-1]):
      best_loss_g = losses_g[-1]
      save_model(model = model["generator"], folder_path = generator_folder_path, name = "generator")
    if(best_loss_d > losses_d[-1]):
      best_loss_d = losses_d[-1]
      save_model(model = model["discriminator"], folder_path = discriminator_folder_path, name = "discriminator")

  return losses_g, losses_d
