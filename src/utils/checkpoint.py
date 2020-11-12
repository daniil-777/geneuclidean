import torch
import pandas as pd
import os

def save_checkpoint_feature(checkpoint_path, idx_max_length, max_length, idx_write):

  state = {'idx_max_length': idx_max_length,
           'max_length': max_length,
           'idx_write': idx_write, 
          }

  torch.save(state, checkpoint_path)

def folds_checkpoint(file_folds_checkpoint_path, type_fold):
    if not os.path.exists(file_folds_checkpoint_path):
        data_folds_checkpoint = pd.DataFrame(columns=['type_fold','fold_no'])
        data_folds_checkpoint.to_csv(file_folds_checkpoint_path, index=False)
    data_checkpoint = pd.read_csv(file_folds_checkpoint_path)
    data_selected = data_checkpoint.loc[(data_checkpoint['type_fold'] == type_fold)]
    if (data_selected.empty):
        data_checkpoint = data_checkpoint.append({'type_fold': type_fold, 'fold_no': 0}, ignore_index=True)
        data_checkpoint.to_csv(file_folds_checkpoint_path, index=False)
        start_idx_fold = 0
    else:
        start_idx_fold = int(data_selected['fold_no'].to_list()[0])

class Checkpoint_Fold():
  def __init__(self, file_folds_checkpoint_path, type_fold):
    self.file_folds_checkpoint_path = file_folds_checkpoint_path
    self.type_fold = type_fold
    if not os.path.exists(self.file_folds_checkpoint_path):
        data_folds_checkpoint = pd.DataFrame(columns=['type_fold','fold_no'])
        data_folds_checkpoint.to_csv(self.file_folds_checkpoint_path, index=False)
    self.data_checkpoint = pd.read_csv(self.file_folds_checkpoint_path)
    self.data_selected = self.data_checkpoint.loc[(self.data_checkpoint['type_fold'] == type_fold)]

  def _get_current_fold(self):
    if (self.data_selected.empty):
      print("yes, empty!!!")
      self.data_checkpoint = self.data_checkpoint.append({'type_fold': self.type_fold, 'fold_no': 0}, ignore_index=True)
      self.data_checkpoint.to_csv(self.file_folds_checkpoint_path, index=False)
      start_idx_fold = 0
    else:
      start_idx_fold = int(self.data_selected['fold_no'].to_list()[0])
    return start_idx_fold
  
  def write_checkpoint(self, idx_fold):
    self.data_checkpoint.loc[(self.data_checkpoint['type_fold'] == self.type_fold), 'fold_no'] = idx_fold
    print("data_selected", self.data_checkpoint)
    self.data_checkpoint.to_csv( self.file_folds_checkpoint_path, index=False)

  
class Checkpoint_Eval():
  def __init__(self, path_checkpoint_evaluator, type_fold, sampling):
    self.path_checkpoint_evaluator = path_checkpoint_evaluator
    self.type_fold = type_fold
    self.sampling = sampling
    if not os.path.exists(self.path_checkpoint_evaluator):
      self.data_checkpoint = pd.DataFrame(columns=['type_fold','sampling','start_rec_fold','start_rec_epoch','start_eval_fold','start_eval_epoch', 'start_pdb'])
      self.data_checkpoint.to_csv(self.path_checkpoint_evaluator, index=False)
    self.data_checkpoint = pd.read_csv(self.path_checkpoint_evaluator)
    self.data_selected = self.data_checkpoint.loc[(self.data_checkpoint['type_fold'] == self.type_fold) & (self.data_checkpoint['sampling'] == self.sampling)]

  def _get_data(self):
    if (self.data_selected.empty):
        # print("empty!!!")
        self.data_checkpoint = self.data_checkpoint.append({'type_fold': self.type_fold, 'sampling': self.sampling, 
                                      'start_rec_fold': 0,'start_rec_epoch': 0,'start_eval_fold': 0,'start_eval_epoch': 0, 'start_pdb': 0}, ignore_index=True)
        self.data_checkpoint.to_csv(self.path_checkpoint_evaluator, index=False)
        self.start_rec_fold = 0
        self.start_rec_epoch = 0
        self.start_eval_fold = 0
        self.start_eval_epoch = 0
    else:   
        self.start_rec_fold = int(self.data_selected['start_rec_fold'].to_list()[0])
        self.start_rec_epoch = int(self.data_selected['start_rec_epoch'].to_list()[0])
        self.start_eval_fold = int(self.data_selected['start_eval_fold'].to_list()[0])
        self.start_eval_epoch = int(self.data_selected['start_eval_epoch'].to_list()[0])
    return self.start_rec_fold, self.start_rec_epoch, self.start_eval_fold, self.start_eval_epoch

  
  def write_record_checkpoint(self, idx_fold, epoch):
    self.data_checkpoint.loc[(self.data_checkpoint['type_fold'] == self.type_fold) & (self.data_checkpoint['sampling'] == self.sampling), 'start_rec_fold'] = idx_fold + 1
    self.data_checkpoint.loc[(self.data_checkpoint['type_fold'] == self.type_fold) & (self.data_checkpoint['sampling'] == self.sampling), 'start_rec_epoch'] = epoch + 1
    self.data_checkpoint.to_csv(self.path_checkpoint_evaluator, index=False)

  def write_eval_checkpoint(self, idx_fold, epoch):
    self.data_checkpoint.loc[(self.data_checkpoint['type_fold'] == self.type_fold) & (self.data_checkpoint['sampling'] == self.sampling), 'start_eval_fold'] = idx_fold + 1
    self.data_checkpoint.loc[(self.data_checkpoint['type_fold'] == self.type_fold) & (self.data_checkpoint['sampling'] == self.sampling), 'start_eval_epoch'] = epoch + 1
    self.data_checkpoint.to_csv(self.path_checkpoint_evaluator, index=False)




