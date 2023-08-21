import torch.nn as nn
import logging, torch
from .Gated_Conv import GatedConv2d, GatedDeConv2d
from  .combined_transformer import combined_Transformer
from .att_block import AttnAware
from einops import rearrange


logger = logging.getLogger(__name__)

class inpaint_model(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, args):
        super().__init__()

        #first three layers 256>128>64>32
        self.pad1 = nn.ReflectionPad2d(3)
        self.act = nn.ReLU(True)
        self.act2 = nn.LeakyReLU(0.2, inplace=True)

        # Gated Conv (encoder)
        self.G_Conv_1 = GatedConv2d(in_channels=4, out_channels=64, kernel_size=7, stride=1, padding=0, activation=args['activation'], norm=args['norm'])
        self.G_Conv_2 = GatedConv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, activation=args['activation'], norm=args['norm'])
        self.G_Conv_3 = GatedConv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, activation=args['activation'], norm=args['norm'])
        self.G_Conv_4 = GatedConv2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1, activation=args['activation'], norm=args['norm'])

        self.G_Conv_5 = GatedConv2d(in_channels=4, out_channels=64, kernel_size=7, stride=1, padding=0, activation=args['activation'], norm=args['norm'])
        self.G_Conv_6 = GatedConv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, activation=args['activation'], norm=args['norm'])
        self.G_Conv_7 = GatedConv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, activation=args['activation'], norm=args['norm'])
        self.G_Conv_8 = GatedConv2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1, activation=args['activation'], norm=args['norm'])

        self.combined_transformer1 = combined_Transformer(args, layer=0)
        self.combined_transformer2 = combined_Transformer(args, layer=1)
        self.combined_transformer3 = combined_Transformer(args, layer=2)
        self.combined_transformer4 = combined_Transformer(args, layer=3)
        self.ln = nn.LayerNorm(32)

        self.att = AttnAware(input_nc=256, activation='gelu', norm='batch', num_heads=16)

        self.RConv1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, dilation=1, stride=1, padding=1)
        self.RConv2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, dilation=1, stride=1, padding=1)
        self.RConv3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, dilation=1, stride=1, padding=1)
        self.RConv4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, dilation=1, stride=1, padding=1)
        self.RConv5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, dilation=1, stride=1, padding=1)
        self.RConv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, dilation=1, stride=1, padding=1)

        self.RConv7 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, dilation=1, stride=1, padding=1)
        self.RConv8 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, dilation=1, stride=1, padding=1)
        self.RConv9 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, dilation=1, stride=1, padding=1)
        self.RConv10 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, dilation=1, stride=1, padding=1)
        self.RConv11 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, dilation=1, stride=1, padding=1)
        self.RConv12 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, dilation=1, stride=1, padding=1)

        # myself use ConvTranspose2d
        self.G_DeConv_1 = GatedDeConv2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1, activation=args['activation'], norm=args['norm'])
        self.G_DeConv_2 = GatedDeConv2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1, activation=args['activation'], norm=args['norm'])
        self.G_DeConv_3 = GatedDeConv2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1, activation=args['activation'], norm=args['norm'])
        self.G_Conv_3_2 = GatedConv2d(in_channels=64, out_channels=3, kernel_size=7, stride=1, padding=0, activation='none', norm='none')

        self.G_DeConv_4 = GatedDeConv2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1, activation=args['activation'], norm=args['norm'])
        self.G_DeConv_5 = GatedDeConv2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1, activation=args['activation'], norm=args['norm'])
        self.G_DeConv_6 = GatedDeConv2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1, activation=args['activation'], norm=args['norm'])
        self.G_Conv_6_2 = GatedConv2d(in_channels=64, out_channels=3, kernel_size=7, stride=1, padding=0, activation='none', norm='none')

        self.batchNorm = nn.BatchNorm2d(256)
        self.padt = nn.ReflectionPad2d(3)
        self.act_last = nn.Sigmoid()    # ZitS first stage

        self.block_size = 32
        self.apply(self._init_weights)

        # calculate parameters
        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):    #https://wandb.ai/wandb_fc/tips/reports/How-to-Initialize-Weights-in-PyTorch--VmlldzoxNjcwOTg1
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv2d, nn.ConvTranspose2d)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, args, new_lr):
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d, torch.nn.ConvTranspose2d)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding, torch.nn.BatchNorm2d)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)


        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert len(
            param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params),)
        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": float(args['weight_decay'])},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=float(new_lr), betas=(0.9, 0.95))
        return optimizer

    def forward(self, img_idx, masks=None):
        ''' Coarse stage '''
        img_idx = img_idx * (1 - masks)
        x = torch.cat((img_idx, masks), dim=1)

        # four layers Gated Conv  (eccoder)
        x = self.pad1(x)
        x1 = self.G_Conv_1(x)
        x2 = self.G_Conv_2(x1)
        x3 = self.G_Conv_3(x2)
        x = self.G_Conv_4(x3)

        # 3 layers residual dilated Conv
        x_residual = x.clone()
        x = self.RConv1(x)
        x = self.act(x)
        x = self.RConv2(x)
        x = self.act(x)
        x = self.RConv3(x) + x_residual
        x = self.act(x)

        # 4 layers shift CSwin transformer
        x_residual = x.clone()
        x, _ = self.combined_transformer1(x)
        x, _ = self.combined_transformer2(x)
        x, _ = self.combined_transformer3(x)
        x, att = self.combined_transformer4(x)
        x = x + x_residual

        # 3 layers residual dilated Conv
        x_residual = x.clone()
        x = self.RConv4(x)
        x = self.act(x)
        x = self.RConv5(x)
        x = self.act(x)
        x = self.RConv6(x) + x_residual
        x = self.act(x)

        # 4 layers Gated DeConv (decoder)
        x = self.G_DeConv_1(x, x3)
        x = self.G_DeConv_2(x, x2)
        x = self.G_DeConv_3(x, x1)
        x = self.padt(x)
        x = self.G_Conv_3_2(x)

        first_out = self.act_last(x)    # Sigmoid

        ''' refine stage '''
        second_input = img_idx * (1 - masks) + first_out * masks
        x = torch.cat((second_input, masks), dim=1)

        # four layers Gated Conv  (eccoder)
        x = self.pad1(x)
        x1 = self.G_Conv_5(x)
        x2 = self.G_Conv_6(x1)
        x3 = self.G_Conv_7(x2)
        x = self.G_Conv_8(x3)

        # second stage uphalf with att
        x_residual = x.clone()
        x = self.RConv7(x)
        x = self.act(x)
        x = self.RConv8(x)
        x = self.act(x)
        x = self.RConv9(x) + x_residual
        x = self.act(x)

        x_att = self.att(x, att=att)
        x_att = self.act(x_att)
        x = x + x_att

        # 3 layers residual dilated Conv
        x_residual = x.clone()
        x = self.RConv10(x)
        x = self.act(x)
        x = self.RConv11(x)
        x = self.act(x)
        x = self.RConv12(x) + x_residual
        x = self.act(x)

        # 4 layers Gated DeConv (decoder)
        x = self.G_DeConv_4(x, x3)
        x = self.G_DeConv_5(x, x2)
        x = self.G_DeConv_6(x, x1)
        x = self.padt(x)
        x = self.G_Conv_6_2(x)

        second_out = self.act_last(x)

        return first_out, second_out


