import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import SAGEConv


class UnetGraphSAGE(nn.Module):
    def __init__(self,input_res,pooling_size, g1, g2,g3,g4,g5, in_feats, h_feats, out_feat, num_step, aggregat):
        super(UnetGraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_feats, int(h_feats / 16), aggregat)
        self.conv2 = SAGEConv(int(h_feats / 16), int(h_feats / 16), aggregat)
        self.conv3 = SAGEConv(int(h_feats / 16), int(h_feats / 8), aggregat)
        self.conv33 = SAGEConv(int(h_feats / 8), int(h_feats / 8), aggregat)

        self.conv4 = SAGEConv(int(h_feats / 8), int(h_feats / 4), aggregat)
        self.conv44 = SAGEConv(int(h_feats / 4), int(h_feats / 4), aggregat)

        self.conv5 = SAGEConv(int(h_feats / 4), int(h_feats / 2), aggregat)
        self.conv55 = SAGEConv(int(h_feats / 2), int(h_feats / 2), aggregat)

        self.conv6 = SAGEConv(int(h_feats / 2), int(h_feats), aggregat)
        self.conv66 = SAGEConv(int(h_feats), int(h_feats), aggregat)


        self.conv7 = SAGEConv(int(h_feats), int(h_feats / 2), aggregat)
        self.conv77 = SAGEConv(int(h_feats / 2), int(h_feats / 2), aggregat)


        self.conv8 = SAGEConv(int(h_feats / 2), int(h_feats / 4), aggregat)
        self.conv88 = SAGEConv(int(h_feats / 4), int(h_feats / 4), aggregat)

        self.conv9 = SAGEConv(int(h_feats / 4), int(h_feats / 8), aggregat)
        self.conv99 = SAGEConv(int(h_feats / 8), int(h_feats / 8), aggregat)


        self.conv10 = SAGEConv(int(h_feats / 8), int(h_feats / 16), aggregat)
        self.conv101 = SAGEConv(int(h_feats / 16), int(h_feats / 16), aggregat)

        self.conv11 = SAGEConv(int(h_feats / 16), out_feat, aggregat)
        self.Maxpool = nn.MaxPool2d((pooling_size, pooling_size), stride=(pooling_size, pooling_size))
        self.Meanpool = nn.AvgPool2d((pooling_size, pooling_size), stride=(pooling_size, pooling_size))
        
        self.upsample1 =nn.ConvTranspose2d(int(h_feats /2 ), int(h_feats / 2), 2, stride=2, padding=0)
        self.upsample2 =nn.ConvTranspose2d(int(h_feats / 4), int(h_feats / 4), 2, stride=2, padding=0)
        self.upsample3 =nn.ConvTranspose2d(int(h_feats / 8), int(h_feats / 8), 2, stride=2, padding=0)
        self.upsample4 =nn.ConvTranspose2d(int(h_feats / 16), int(h_feats / 16), 2, stride=2, padding=0)

        self.g1 = g1
        self.g2 = g2
        self.g3 = g3
        self.g4 = g4
        self.g5 = g5

        self.num_step = num_step
        # self.get_graph=get_graph
        self.input_res=input_res
        self.pooling_size=pooling_size

    def forward(self, in_feat, exteraVar1):

            h1 = F.relu(self.conv1(self.g1, in_feat))

            h22 = F.relu(self.conv2(self.g1, h1))
            h2=h22.view(6, self.input_res, self.input_res, -1)
            h2=torch.permute(h2, (3, 0 , 1, 2))
            h2=self.Meanpool(h2).view(-1, int(6*self.input_res/self.pooling_size*self.input_res/self.pooling_size))
            h2=torch.transpose(h2, 0 , 1)
            # g2=self.get_graph(24)

            h3 = F.relu(self.conv3(self.g2, h2))
            h33 = F.relu(self.conv33(self.g2, h3))
            h3=h33.view(6, int(self.input_res/self.pooling_size), int(self.input_res/self.pooling_size), -1)
            h3=torch.permute(h3, (3, 0 , 1, 2))
            h3=self.Meanpool(h3).view(-1, int(6*self.input_res/(self.pooling_size)**2*self.input_res/(self.pooling_size)**2))
            h3=torch.transpose(h3, 0 , 1)
            # g3=self.get_graph(self.input_res/(self.pooling_size)**2)

            h4 = F.relu(self.conv4(self.g3, h3))
            h44 = F.relu(self.conv44(self.g3, h4))
            h4=h44.view(6,int(self.input_res/(self.pooling_size)**2),int(self.input_res/(self.pooling_size)**2),-1)
            h4=torch.permute(h4, (3, 0 , 1, 2))
            h4=self.Meanpool(h4).view(-1, int(6*self.input_res/(self.pooling_size)**3*self.input_res/(self.pooling_size)**3))
            h4=torch.transpose(h4, 0 , 1)
            # g4=self.get_graph(self.input_res/(self.pooling_size)**3)

            h5 = F.relu(self.conv5(self.g4, h4))
            h55 = F.relu(self.conv55(self.g4, h5))
            h5 = h55.view(6,int(self.input_res/(self.pooling_size)**3),int(self.input_res/(self.pooling_size)**3),-1)
            h5=torch.permute(h5, (3, 0 , 1, 2))
            h5=self.Meanpool(h5).view(-1, int(6*self.input_res/(self.pooling_size)**4*self.input_res/(self.pooling_size)**4))
            h5=torch.transpose(h5, 0 , 1)

            h6 = F.relu(self.conv6(self.g5, h5))
            h6 = F.relu(self.conv66(self.g5, h6))
            h6 = F.relu(self.conv7(self.g5, h6)).view(6,int(self.input_res/(self.pooling_size)**4),int(self.input_res/(self.pooling_size)**4),-1)
            h6=torch.permute(h6, (0, 3 , 1, 2))
            h6=self.upsample1(h6)
            h6=torch.permute(h6, (1, 0 , 2, 3)).reshape(-1, int(6*self.input_res/(self.pooling_size)**3*self.input_res/(self.pooling_size)**3))
            h6=torch.transpose(h6, 0 , 1)            
            h6 = torch.cat((h6, h55), dim=1)

            h6 = F.relu(self.conv7(self.g4, h6))
            h6 = F.relu(self.conv77(self.g4, h6))
            h6 = F.relu(self.conv8(self.g4, h6)).view(6,int(self.input_res/(self.pooling_size)**3),int(self.input_res/(self.pooling_size)**3),-1)
            h6=torch.permute(h6, (0, 3 , 1, 2))
            h6=self.upsample2(h6)
            h6=torch.permute(h6, (1, 0 , 2, 3)).reshape(-1, int(6*self.input_res/(self.pooling_size)**2*self.input_res/(self.pooling_size)**2))
            h6=torch.transpose(h6, 0 , 1)
            h6 = torch.cat((h6, h44), dim=1)


            h6 = F.relu(self.conv8(self.g3, h6))
            h6 = F.relu(self.conv88(self.g3, h6))
            h6 = F.relu(self.conv9(self.g3, h6)).view(6,int(self.input_res/(self.pooling_size)**2),int(self.input_res/(self.pooling_size)**2),-1)
            h6=torch.permute(h6, (0, 3 , 1, 2))
            h6=self.upsample3(h6)
            h6=torch.permute(h6, (1, 0 , 2, 3)).reshape(-1, int(6*self.input_res/(self.pooling_size)*self.input_res/(self.pooling_size)))
            h6=torch.transpose(h6, 0 , 1)
            h6 = torch.cat((h6, h33), dim=1)



            h6 = F.relu(self.conv9(self.g2, h6))
            h6 = F.relu(self.conv99(self.g2, h6))
            h6 = F.relu(self.conv10(self.g2, h6)).view(6,int(self.input_res/(self.pooling_size)),int(self.input_res/(self.pooling_size)),-1)
            h6=torch.permute(h6, (0, 3 , 1, 2))
            h6=self.upsample4(h6)
            h6=torch.permute(h6, (1, 0 , 2, 3)).reshape(-1, int(6*self.input_res*self.input_res))
            h6=torch.transpose(h6, 0 , 1)
            h6 = torch.cat((h6, h22), dim=1)




            h6 = F.relu(self.conv10(self.g1, h6))
            h6 = F.relu(self.conv101(self.g1, h6))            
            out = self.conv11(self.g1, h6)
            return out
