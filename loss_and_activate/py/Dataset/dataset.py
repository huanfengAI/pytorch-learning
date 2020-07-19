from torch.utils.data import Dataset
class myDataset(Dataset): 
  def __init__ (self, csv_file, txt_file, root_dir, other_file) : 
     self.csv_data = pd.read_csv(csv_file) 
     with open(txt_file, 'r' ) as f: 
        data_list = f.readlines( ) 
     self.txt_data = data_list 
     self.root_dir = root_dir 

  def __len__ (self): 
    return len(self.csv_data) 
  
  def __getitem__ (self,idx) : 
   data = (self. csv_data [idx],self.txt_data[idx]) 
   return data
  #数据集构建的时候，要继承Dataset，然后要写三个方法，
  # 第一个方法是__init__方法
  #第二个方法是__len__
  #第三个方法是__gettem__
  #第一个方法我们要完成将传入的数据集的样本和标签分别赋给两个属性
  #第二个方法我们要返回数据集的长度
  #第三个方法（参数索引）是通过操作第一个方法设置的属性，返回一个样本
