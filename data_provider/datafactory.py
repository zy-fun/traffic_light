from torch.utils.data import DataLoader
from dataloader import Car_Passing_Data

def data_provider(args, flag: str="train"):
    batch_size = args.batch_size
    shuffle_flag = False if (flag == 'test' or flag == 'TEST') else True
    drop_last = True if flag == 'train' else False

    data_set = Car_Passing_Data(flag=flag)
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        # num_workers=args.num_workers,
        drop_last=drop_last)

    return data_set, data_loader