import torch
from torch import nn
#
from .mlp import MultiLayerPerceptron

# 在这里自己定义一个CNN神经网络
class CNN(nn.Module):
    def __init__(self):
        self.input_dim =5
        self.input_len =336
        self.output_len =32
        super(CNN, self).__init__()
        self.conv1= nn.Sequential(
            nn.Conv2d(
                in_channels=self.input_dim * self.input_len,
                out_channels = self.output_len,

            ))
    def forward(self,x):
        x= self.conv1(x)
        return x


class STID(nn.Module):
    """
    The implementation of CIKM 2022 short paper
        "Spatial-Temporal Identity: A Simple yet Effective Baseline for Multivariate Time Series Forecasting"
    Link: https://arxiv.org/abs/2208.05233
    """

    def __init__(self, **model_args):
        super().__init__()
        # attributes
        self.num_nodes = model_args["num_nodes"]
        self.node_dim = model_args["node_dim"]
        self.input_len = model_args["input_len"]
        self.input_dim = model_args["input_dim"]
        self.embed_dim = model_args["embed_dim"]
        self.output_len = model_args["output_len"]
        self.num_layer = model_args["num_layer"]
        self.temp_dim_tid = model_args["temp_dim_tid"]
        self.temp_dim_diw = model_args["temp_dim_diw"]
        self.time_of_day_size = model_args["time_of_day_size"]
        self.day_of_week_size = model_args["day_of_week_size"]

        self.if_time_in_day = model_args["if_T_i_D"]
        self.if_day_in_week = model_args["if_D_i_W"]
        self.if_spatial = model_args["if_node"]

        # spatial embeddings
        if self.if_spatial:
            self.node_emb = nn.Parameter(
                torch.empty(self.num_nodes, self.node_dim))
            nn.init.xavier_uniform_(self.node_emb)
        # temporal embeddings
        if self.if_time_in_day:
            self.time_in_day_emb = nn.Parameter(
                torch.empty(self.time_of_day_size, self.temp_dim_tid))
            nn.init.xavier_uniform_(self.time_in_day_emb)
        if self.if_day_in_week:
            self.day_in_week_emb = nn.Parameter(
                torch.empty(self.day_of_week_size, self.temp_dim_diw))
            nn.init.xavier_uniform_(self.day_in_week_emb)
        #torch.backends.cudnn.benchmark = False
        # embedding layer


        # encoding
        self.hidden_dim = self.embed_dim+self.node_dim * \
            int(self.if_spatial)+self.temp_dim_tid*int(self.if_day_in_week) + \
            self.temp_dim_diw*int(self.if_time_in_day)
        self.encoder = nn.Sequential(
            *[MultiLayerPerceptron(self.hidden_dim, self.hidden_dim) for _ in range(self.num_layer)])

        # regression
        self.regression_layer = nn.Conv2d(
            in_channels=self.hidden_dim, out_channels=self.output_len, kernel_size=(1, 1), bias=True)
        # self.time_series_emb_layer = nn.Conv2d(
        #      in_channels=self.input_dim * self.input_len, out_channels=self.embed_dim, kernel_size=(1, 1), bias=True)
        self.time_series_emb_layer = nn.Conv2d(
             in_channels=self.input_dim * self.input_len, out_channels=self.embed_dim, kernel_size=(1, 1), bias=True)


        # self.time_series_emb_layer = nn.Linear(in_features=self.input_dim * self.input_len, out_features=self.embed_dim,
        #                                        bias=True)

        # self.time_layer = nn.Conv2d(self.input_dim * self.input_len,self.embed_dim)

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, train: bool, **kwargs) -> torch.Tensor:
        """Feed forward of STID.

        Args:
            history_data (torch.Tensor): history data with shape [B, L, N, C]

        Returns:
            torch.Tensor: prediction wit shape [B, L, N, C]
        """
        print("history_data")
        print(history_data.shape)
        # prepare data
        input_data = history_data[..., range(self.input_dim)]

        # "time_of_day_size": 24,
        # "day_of_week_size": 7,
        # "day_of_month_size": 31,
        # "day_of_year_size": 366
        if self.if_day_in_week:
            d_i_w_data = history_data[..., 2]
            print("d_i_w_data")
            print(d_i_w_data.shape)                                        #torch.Size([64, 336, 7])
            print("d_i_w_data[:, -1, :]")
            print(d_i_w_data[:, -1, :].shape)                              #torch.Size(64,7)
            # 执行到这里就报错了，d_i_w_data[:, -1, :] 的shape 是(64,7)
            #self.day_in_week_emb的shape 是(7,32)         (64,7)
            day_in_week_emb = self.day_in_week_emb[(d_i_w_data[:, -1, :]).type(torch.LongTensor)]
            print("day_in_week_emb")
            print(day_in_week_emb.shape)
        else:
            day_in_week_emb = None

        if self.if_time_in_day:
            t_i_d_data = history_data[..., 1]
            # In the datasets used in STID, the time_of_day feature is normalized to [0, 1]. We multiply it by 288 to get the index.
            # If you use other datasets, you may need to change this line.
            print("self.time_in_day_emb")
            print(self.time_in_day_emb.shape)                               # (288,32)     ([288, 32])           [24, 32]
            print("self.t_i_d_data")
            print(t_i_d_data.shape)  # (32,48,321)                          （32，12，358） ([32, 48, 321])       [64, 336, 7]
            a=t_i_d_data[:, -1, :] * self.time_of_day_size  #               （32，358）     ([32, 321])           [64, 7]
            print("a的shape")
            print(a.shape)          # [24, 32]        [64, 7]
            time_in_day_emb = self.time_in_day_emb[(t_i_d_data[:, -1, :] * self.time_of_day_size).type(torch.LongTensor)] #[64, 7, 32]
            print("time_in_day_emb的shape")                                 #（32,358,32）
            print(time_in_day_emb.shape)
        else:
            time_in_day_emb = None

        # time series embedding
        batch_size, _, num_nodes, _ = input_data.shape
        input_data = input_data.transpose(1, 2).contiguous()
        input_data = input_data.view(
            batch_size, num_nodes, -1).transpose(1, 2).unsqueeze(-1)



        node_emb = []
        if self.if_spatial:
            # expand node embeddings
            node_emb.append(self.node_emb.unsqueeze(0).expand(
                batch_size, -1, -1).transpose(1, 2).unsqueeze(-1))
        # temporal embeddings
        tem_emb = []
        if time_in_day_emb is not None:
            tem_emb.append(time_in_day_emb.transpose(1, 2).unsqueeze(-1))  #time_in_day_emb ： [64, 7, 32]  torch.Size([64, 32, 7,1])
        if day_in_week_emb is not None:
            tem_emb.append(day_in_week_emb.transpose(1, 2).unsqueeze(-1)) #day_in_week_emb [64, 7, 32])   torch.Size([64, 32, 7,1])
        # CNN1 = CNN().cuda()
        # time_series_emb = CNN1(input_data)
        # concate all embeddings
        time_series_emb = self.time_series_emb_layer(input_data) # 这里报错
        # time_series_emb = self.time_series_emb_layer(torch.squeeze(input_data, dim=3).transpose(1, 2))
        # time_series_emb = self.time_layer(torch.permute(input_data,(0,3,2,1)))
        # input_dim = 5
        # embed_dim = 32
        # input_len = 336
        # time_layer = torch.nn.Linear(input_dim * input_len, embed_dim)
        # time_layer =time_layer.cuda()
        # input_data= torch.permute(input_data, (0, 3, 2, 1))
        # input_data = input_data.cuda()
        # time_series_emb = time_layer(input_data)
        # time_series = torch.permute(time_series_emb, (0, 3, 2, 1))
        hidden = torch.cat([time_series_emb] + node_emb + tem_emb, dim=1)
        # encoding
        hidden = self.encoder(hidden)

        # regression
        prediction = self.regression_layer(hidden)

        return prediction