import os, sys, torch, logging, yaml, time
import random
import colorful as c
import torchvision
from torchsummary import summary
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from warmup_scheduler import GradualWarmupScheduler
from my_utils import set_seed, Visualization_of_training_results, SSIM, PSNR, LPIPS, LPIPS_SET, Visualization_of_training_results_val
from models.model import inpaint_model
from models.Discriminator import NLayerDiscriminator
from dataset import get_dataset
from tqdm import tqdm
from losses.L1 import l1_loss
from losses.Perceptual import PerceptualLoss
from losses.Adversarial import NonSaturatingWithR1
from losses.HSV_color import HSV
from losses.Style import StyleLoss
from tensorboardX import SummaryWriter

# set seed
set_seed(731)
gpu = torch.device("cuda")

# open config file
with open('./config/model_config.yml', 'r') as config:
    args = yaml.safe_load(config)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

# set log
log_path = args['ckpt_path'] + args['name']
if not os.path.isdir(log_path):
    os.makedirs(log_path)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
sh = logging.StreamHandler(stream=sys.stdout)
logger.addHandler(sh)
logger.propagate = False
fh = logging.FileHandler(os.path.join(log_path, args['name'] + '.txt'))
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)

# set tensorboardx log
log_dir = os.path.join(args['ckpt_path'], args['name'], 'log')
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir=log_dir, filename_suffix=f'_'+args['name'])

# Define the model
Inpaint_model = inpaint_model(args)
Discriminator = NLayerDiscriminator(input_nc=3)

# Define the dataset
train_dataset = get_dataset(args['data_path'], mask_path=args['mask_path'], is_train=True,image_size=args['image_size'])

test_dataset = get_dataset(args['val_path'], test_mask_path=args['val_mask_path'],is_train=False, image_size=args['image_size'])

# set initial
previous_epoch = -1
bestAverageF1 = 0
iterations = 0
best_psnr = 0
best_ssim = 0
best_lpips = 1
best_epoch_psnr = 0
best_epoch_ssim = 0
best_epoch_lpips = 0
Total_time = []

# loaded_ckpt
if os.path.exists(args['resume_ckpt']):
    data = torch.load(args['resume_ckpt'])
    D_data = torch.load(args['resume_D_ckpt'])
    Inpaint_model.load_state_dict(data['state_dict'],strict=False)
    Discriminator.load_state_dict(D_data['discriminator'])
    Inpaint_model = Inpaint_model.to(args['gpu'])
    Discriminator = Discriminator.to(args['gpu'])
    logger.info('Finished reloading the Epoch '+c.yellow(str(data['epoch']))+' model')
    # Optimizer
    raw_model = Inpaint_model.module if hasattr(Inpaint_model, "module") else Inpaint_model  # 要問一下
    optimizer = raw_model.configure_optimizers(args, new_lr=args['learning_rate'])
    # update optimizer info
    optimizer.load_state_dict(data['optimizer'])
    D_optimizer = optim.Adam(Discriminator.parameters(), lr=args['D_learning_rate'], betas=(0.9, 0.95))
    D_optimizer.load_state_dict(D_data['D_optimizer'])
    iterations = data['iterations']
    #bestAverageF1 = data['best_validation']
    previous_epoch = data['epoch']
    logger.info('Finished reloading the Epoch '+c.yellow(str(data['epoch']))+' optimizer')
    logger.info(c.blue('------------------------------------------------------------------'))
    logger.info('resume training with iterations: '+c.yellow(str(iterations))+ ', previous_epoch: '+c.yellow(str(previous_epoch)))
    logger.info(c.blue('------------------------------------------------------------------'))
else:
    # start form head
    Inpaint_model = Inpaint_model.to(args['gpu'])
    Discriminator = Discriminator.to(args['gpu'])
    logger.info(c.blue('------------------------------------------------------------------'))
    logger.info(c.red('Warnning: There is no trained model found. An initialized model will be used.'))
    logger.info(c.red('Warnning: There is no previous optimizer found. An initialized optimizer will be used.'))
    logger.info(c.blue('------------------------------------------------------------------'))
    # Optimizer
    raw_model = Inpaint_model.module if hasattr(Inpaint_model, "module") else Inpaint_model #要問一下
    optimizer = raw_model.configure_optimizers(args, new_lr=args['learning_rate'] )
    D_optimizer = optim.Adam(Discriminator.parameters(), lr=args['D_learning_rate'], betas=(0.9, 0.95))

if args['lr_decay']:
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, args['train_epoch'] - args['warmup_epoch'],
                                                            eta_min=float(args['lr_min']))
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=args['warmup_epoch'],
                                       after_scheduler=scheduler_cosine)
    # scheduler.step()

    D_scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(D_optimizer, args['train_epoch'] - args['D_warmup_epoch'],
                                                            eta_min=float(args['D_lr_min']))
    D_scheduler = GradualWarmupScheduler(D_optimizer, multiplier=1, total_epoch=args['D_warmup_epoch'],
                                       after_scheduler=D_scheduler_cosine)
    # D_scheduler.step()

    for i in range(1, previous_epoch):  #如果有預訓練黨要step到該階段
        scheduler.step()
        D_scheduler.step()

# DataLoaders
train_loader = DataLoader(train_dataset, shuffle=True, pin_memory=True,
                                  batch_size=args['batch_size'] // args['world_size'],  # BS of each GPU
                                  num_workers=args['num_workers'])

test_loader = DataLoader(test_dataset, shuffle=False, pin_memory=True,
                         batch_size=args['batch_size'] // args['world_size'],
                         num_workers=args['num_workers'])

# Load pre_trained VGG model (for Perceptual_loss)
if args['Lambda_Perceptual'] is not 0 or not None:
    logger.info('Try load VGG mdoel for Perceptual_loss')
    VGG = PerceptualLoss()
if VGG is None:
    logger.info(c.red('Warnning: There is no pre_trained VGG model found. Try again or Check the Internet.'))
else:
    logger.info(c.blue('-----------------------------')+c.magenta(' Loaded! ')+c.blue('----------------------------'))
if args['Lambda_Style'] is not 0 or not None:
    logger.info('Try load VGG mdoel for Style_loss')
    style = StyleLoss()
if VGG is None:
    logger.info(c.red('Warnning: There is no pre_trained VGG model found. Try again or Check the Internet.'))
else:
    logger.info(c.blue('-----------------------------')+c.magenta(' Loaded! ')+c.blue('----------------------------'))

adv = NonSaturatingWithR1()

logger.info('Try load alex mdoel for LPIPS')
alex = LPIPS_SET()
alex = alex.to(args['gpu'])
if alex is None:
    logger.info(c.red('Warnning: There is no pre_trained alex model found. Try again or Check the Internet.'))
else:
    logger.info(c.blue('-----------------------------')+c.magenta(' Loaded! ')+c.blue('----------------------------'))

print('==> Training start: ')
summary(raw_model, [(3, 256, 256), (1, 256, 256)])  # won't write in log (model show)


for epoch in range(args['train_epoch']):
    if previous_epoch != -1 and epoch < previous_epoch:
        continue
    if epoch == previous_epoch + 1:
        logger.info("Resume from Epoch %d" % epoch)

    epoch_start = time.time()
    raw_model.train()   # Train MODE
    Discriminator.train()
    loader = train_loader

    for it, items in enumerate(tqdm(loader, disable=False)):
        for k in items:
            if type(items[k]) is torch.Tensor:
                items[k] = items[k].to(args['gpu']).requires_grad_()    # img mask edge name

       # G
        raw_model.zero_grad()
        Discriminator.zero_grad()

        first_out, second_out = raw_model(items['img'], items['mask'])
        D_pred_img, _ = Discriminator(second_out)

        L1_loss_1 = l1_loss(items['img'], first_out, Edge=None)
        L1_loss_2 = l1_loss(items['img'], second_out, Edge=None)
        L1_loss = args['Lambda_L1'] * (L1_loss_1 + L1_loss_2)

        Perceptual_loss_1 = VGG.forward(items['img'], first_out, mask=items['mask'])
        Perceptual_loss_2 = VGG.forward(items['img'], second_out, mask=items['mask'])
        Perceptual_loss = args['Lambda_Perceptual'] * (Perceptual_loss_1 + Perceptual_loss_2)

        style_loss_1 = style(first_out, items['img'])
        style_loss_2 = style(second_out, items['img'])
        style_loss = args['Lambda_Style'] * (style_loss_1 + style_loss_2)

        # HSV_loss_H_1, HSV_loss_S_1, HSV_loss_V_1, HSV_loss_1 = HSV(items['img'], first_out, edge=None)
        HSV_loss_H_2, HSV_loss_S_2, HSV_loss_V_2, HSV_loss_2 = HSV(items['img'], second_out, edge=None)
        HSV_loss = args['Lambda_HSV'] * HSV_loss_2

        LG = adv.generator_loss(D_pred_img, mask=None)
        LG = args['Lambda_LG'] * LG

        G_loss = L1_loss + Perceptual_loss + style_loss + HSV_loss + LG

        G_loss.backward()
        optimizer.step()

        # D
        raw_model.zero_grad()
        Discriminator.zero_grad()

        first_out, second_out = raw_model(items['img'], items['mask'])
        D_img, _ = Discriminator(items['img'])
        D_pred_img_2, _ = Discriminator(second_out.detach_())
        LD1 = adv.discriminator_real_loss(items['img'], D_img)  # LD1+LGP (real loss)
        LD2 = adv.discriminator_fake_loss(D_pred_img_2, mask=items['mask'])  # LD2+3 (fake loss)

        D_loss = args['Lambda_LD1'] * LD1 + args['Lambda_LD2'] * LD2
        D_loss.backward()
        D_optimizer.step()

        iterations += 1

        if iterations % 500 == 0:
            Visualization_of_training_results(first_out, second_out, items['img'], items['mask'], log_path, iterations)
            writer.add_scalar('loss/LD1', LD1.item(), iterations)  # tensorboard for log (visualize)
            writer.add_scalar('loss/LD2', LD2.item(), iterations)  # tensorboard for log (visualize)
            writer.add_scalar('Total_loss/D_total_loss', D_loss.item(), iterations)  # tensorboard for log (visualize)
            writer.add_scalar('loss/L1_loss_1', L1_loss_1.item(), iterations)  # tensorboard for log (visualize)
            writer.add_scalar('loss/L1_loss_2', L1_loss_2.item(), iterations)  # tensorboard for log (visualize)
            writer.add_scalar('loss/Perceptual_loss_1', Perceptual_loss_1.item(), iterations)
            writer.add_scalar('loss/Perceptual_loss_2', Perceptual_loss_2.item(), iterations)
            writer.add_scalar('loss/style_loss_1', style_loss_1.item(), iterations)
            writer.add_scalar('loss/style_loss_2', style_loss_2.item(), iterations)
            writer.add_scalar('loss/LG', LG.item(), iterations)
            writer.add_scalar('loss/HSV_loss', HSV_loss_2.item(), iterations)
            writer.add_scalar('Total_loss/G_total_loss', G_loss.item(), iterations)

    logger.info(
        f"| first stage |: L1_loss {L1_loss_1.item():.5f}. | Perceptual_loss {Perceptual_loss_1.item():.5f}. | "
        f"style_loss {style_loss_1.item():.5f} ")
    logger.info(
        f"| second stage |: L1_loss {L1_loss_2.item():.5f}. | Perceptual_loss {Perceptual_loss_2.item():.5f}. | "
        f"style_loss {style_loss_2.item():.5f}. | HSV_loss {HSV_loss_2.item():.5f}.")
    logger.info(
        f"epoch {epoch + 1} iter {iterations}: D1_loss {LD1.item():.5f}. | D2_loss {LD2.item():.5f}. | G_loss {LG.item():.5f}. | "
        f"{c.magenta('G_Total_loss')} {G_loss.item():.5f}. | {c.magenta('D_Total_loss')} {D_loss.item():.5f}. | lr {scheduler.get_lr()[0]:e}")

    # start EVAL
    logger.info(c.blue('-----------------------------')+c.cyan(' Start EVAL! ')+c.blue('-------------------------------------'))
    # Evaluation (Validation)
    raw_model.eval()    # eval MODE
    Discriminator.eval()
    loader = test_loader
    PSNR_center = []
    SSIM_center = []
    LPIPS_center = []
    PSNR_center1 = []
    SSIM_center1 = []
    LPIPS_center1 = []
    val_rand = random.randrange(int(len(open(args['val_path'], 'rU').readlines())/ args['batch_size']))
    val_int = 0

    for val_it, val_items in enumerate(loader):
        for k in val_items:
            if type(val_items[k]) is torch.Tensor:
                val_items[k] = val_items[k].to(args['gpu'])

        # in to the model (no_grad)
        with torch.no_grad():
            raw_model.zero_grad()
            Discriminator.zero_grad()

            first_out, second_out = raw_model(val_items['img'], val_items['mask'])

            L1_loss_1 = l1_loss(val_items['img'], first_out, Edge=None)
            L1_loss_2 = l1_loss(val_items['img'], second_out, Edge=None)
            L1_loss = args['Lambda_L1'] * (L1_loss_1 + L1_loss_2)

            Perceptual_loss_1 = VGG.forward(val_items['img'], first_out, mask=val_items['mask'])
            Perceptual_loss_2 = VGG.forward(val_items['img'], second_out, mask=val_items['mask'])
            Perceptual_loss = args['Lambda_Perceptual'] * (Perceptual_loss_1 + Perceptual_loss_2)

            style_loss_1 = style(first_out, val_items['img'])
            style_loss_2 = style(second_out, val_items['img'])
            style_loss = args['Lambda_Style'] * (style_loss_1 + style_loss_2)

            # HSV_loss_H_1, HSV_loss_S_1, HSV_loss_V_1, HSV_loss_1 = HSV(items['img'], first_out, edge=None)
            HSV_loss_H_2, HSV_loss_S_2, HSV_loss_V_2, HSV_loss_2 = HSV(val_items['img'], second_out, edge=None)
            HSV_loss = args['Lambda_HSV'] * HSV_loss_2

            G_loss = L1_loss + Perceptual_loss + style_loss + HSV_loss


            PSNR_center1.append(PSNR(val_items['img'], first_out))
            SSIM_center1.append(SSIM(val_items['img'], first_out))
            LPIPS_center1.append(LPIPS(val_items['img'], first_out, alex))
            PSNR_center.append(PSNR(val_items['img'], second_out))
            SSIM_center.append(SSIM(val_items['img'], second_out))
            LPIPS_center.append(LPIPS(val_items['img'], second_out, alex))
            if val_rand == val_int:
                Visualization_of_training_results_val(first_out, second_out, val_items['img'], val_items['mask'], log_path,
                                                  iterations)
            val_int+=1

    writer.add_scalar('val_loss/L1_loss_1', L1_loss_1.item(), iterations)
    writer.add_scalar('val_loss/L1_loss_2', L1_loss_2.item(), iterations)
    writer.add_scalar('val_loss/Perceptual_loss_1', Perceptual_loss_1.item(), iterations)
    writer.add_scalar('val_loss/Perceptual_loss_2', Perceptual_loss_2.item(), iterations)
    writer.add_scalar('val_loss/style_loss_1', style_loss_1.item(), iterations)
    writer.add_scalar('val_loss/style_loss_2', style_loss_2.item(), iterations)
    writer.add_scalar('val_loss/HSV_loss', HSV_loss_2.item(), iterations)
    writer.add_scalar('val_loss/G_total_loss', G_loss.item(), iterations)

    PSNR_center1 = torch.stack(PSNR_center1).mean().item()
    SSIM_center1 = torch.stack(SSIM_center1).mean().item()
    LPIPS_center1 = torch.stack(LPIPS_center1).mean().item()
    PSNR_center = torch.stack(PSNR_center).mean().item()
    SSIM_center = torch.stack(SSIM_center).mean().item()
    LPIPS_center = torch.stack(LPIPS_center).mean().item()

    writer.add_scalar('Evaluation_Metrics1/PSNR_center', PSNR_center1, epoch + 1)
    writer.add_scalar('Evaluation_Metrics1/SSIM_center', SSIM_center1, epoch + 1)
    writer.add_scalar('Evaluation_Metrics1/LPIPS_center', LPIPS_center1, epoch + 1)
    writer.add_scalar('Evaluation_Metrics2/PSNR_center', PSNR_center, epoch + 1)
    writer.add_scalar('Evaluation_Metrics2/SSIM_center', SSIM_center, epoch + 1)
    writer.add_scalar('Evaluation_Metrics2/LPIPS_center', LPIPS_center, epoch + 1)

    # Save ckpt (best PSNR)
    if PSNR_center > best_psnr:
        best_psnr = PSNR_center
        best_epoch_psnr = epoch
        torch.save({'epoch': epoch + 1,
                    'state_dict': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'iterations': iterations,
                    'best_PSNR': PSNR_center,
                    'best_SSIM': SSIM_center,
                    'best_LPIPS': LPIPS_center
                    }, os.path.join(log_path, "model_bestPSNR.pth"))

    # Save ckpt (best SSIM)
    if SSIM_center > best_ssim:
        best_ssim = SSIM_center
        best_epoch_ssim = epoch
        torch.save({'epoch': epoch + 1,
                    'state_dict': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'iterations': iterations,
                    'best_PSNR': PSNR_center,
                    'best_SSIM': SSIM_center,
                    'best_LPIPS': LPIPS_center
                    }, os.path.join(log_path, "model_bestSSIM.pth"))

    # Save ckpt (best LPIPS)
    if LPIPS_center < best_lpips:
        best_lpips = LPIPS_center
        best_epoch_lpips = epoch
        torch.save({'epoch': epoch + 1,
                    'state_dict': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'iterations': iterations,
                    'best_PSNR': PSNR_center,
                    'best_SSIM': SSIM_center,
                    'best_LPIPS': LPIPS_center
                    }, os.path.join(log_path, "model_bestLPIPS.pth"))

    # Save ckpt (each EPOCH)
    torch.save({'epoch': epoch + 1,
                'state_dict': raw_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'iterations': iterations,
                'best_PSNR': PSNR_center,
                'best_SSIM': SSIM_center,
                'best_LPIPS': LPIPS_center
                }, os.path.join(log_path, "model_" + str(epoch + 1) + ".pth"))

    logger.info(f"| first stage | current_PSNR {PSNR_center1:.5f}. | current_SSIM {SSIM_center1:.5f}. | current_LPIPS {LPIPS_center1:.5f}. ")
    logger.info(f"current_epoch {epoch + 1} : current_PSNR {PSNR_center:.5f}. | "
                f"best_epoch {best_epoch_psnr+1} : best_PSNR {best_psnr:.5f}."
                f"\ncurrent_epoch {epoch + 1} : current_SSIM {SSIM_center:.5f}.  | "
                f"best_epoch {best_epoch_ssim+1} : best_SSIM {best_ssim:.5f}."
                f"\ncurrent_epoch {epoch + 1} : current_LPIPS {LPIPS_center:.5f}. | "
                f"best_epoch {best_epoch_lpips+1} : best_LPIPS {best_lpips:.5f}.")

    # Save D_ckpt (each EPOCH)
    torch.save({'discriminator': Discriminator.state_dict(),
                'D_optimizer': D_optimizer.state_dict(),
                }, os.path.join(log_path, "Discriminator_" + str(epoch + 1) + ".pth"))

    # Save the last model
    torch.save({'epoch': epoch + 1,
                'state_dict': raw_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'iterations': iterations,
                'best_PSNR': PSNR_center,
                'best_SSIM': SSIM_center,
                'best_LPIPS': LPIPS_center
                }, os.path.join(log_path, "model_last.pth"))

    # Save D_ckpt (last EPOCH)
    torch.save({'discriminator': Discriminator.state_dict(),
                'D_optimizer': D_optimizer.state_dict(),
                }, os.path.join(log_path, "Discriminator_last.pth"))

    if args['lr_decay']:
        scheduler.step()
        D_scheduler.step()
        writer.add_scalar('lr/G_lr', scheduler.get_lr()[0], epoch + 1)
        writer.add_scalar('lr/D_lr', D_scheduler.get_lr()[0], epoch + 1)

    epoch_time = (time.time() - epoch_start)  # teain one epoch time
    logger.info(f"This epoch cost {epoch_time:.5f} seconds!")
    Total_time.append(epoch_time)
    logger.info(c.blue('------------------------------')+c.cyan(' End EVAL! ')+c.blue('--------------------------------------'))

writer.close()
Total_time = sum(Total_time)
logger.info(f"\n --------------------- Total time cost {Total_time/60/60:.5f} HR! ----------------------"
            f"\n -------------------------- ALL TRAINING IS DONE!! --------------------------")
