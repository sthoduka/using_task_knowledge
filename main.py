import pytorch_lightning as pl
from argparse import ArgumentParser
import trainer
import torch
import pdb
from pytorch_lightning.callbacks import ModelCheckpoint

def main():
    parser = ArgumentParser()

    parser.add_argument('--data_root', default='', type=str, help='Root path of input videos')
    parser.add_argument('--dataset', default='', type=str, help='armbench,failure,imperfect_pour')
    parser.add_argument('--training_trials', default='train.json', type=str, help='JSON file with training trials for armbench')
    parser.add_argument('--val_trials', default='test.json', type=str, help='JSON file with val trials for armbench')
    parser.add_argument('--test_trials', default='test.json', type=str, help='JSON file with test trials for armbench')

    # dataset specific
    parser.add_argument('--num_classes', default=2, type=int, help='number of classes')

    parser.add_argument('--action_subset_frame_selection', action='store_true', help="true if limiting range for frame selection")
    parser.add_argument('--defect_segment_selection', action='store_true', help="true if selecting frames based on time when failures occur")
    parser.add_argument('--cropped', action='store_true', help="true if images should be cropped with fixed values")
    parser.add_argument('--action_crop', action='store_true', help="true if images should be cropped based on the action")
    parser.add_argument('--random_frame_select', action='store_true', help="select random frames during training")
    parser.add_argument('--dense_sampling', action='store_true', help="sample all frames")
    parser.add_argument('--variable_frame_rate_augmentation', action='store_true', help="apply variable frame rate augmentation")
    parser.add_argument('--non_action_aligned_variable_frame_rate_augmentation', action='store_true', help="apply augmentation randomly without aligning to actions")

    parser.add_argument('--num_outcome_classes', default=2, type=int, help="number of classification outcomes")
    parser.add_argument('--selected_action', default='', type=str, help="only focus on this action")
    parser.add_argument('--actions_separately', action='store_true', help="true if we want to treat each action within the task as a separate trial")

    # model specific
    parser.add_argument('--mvit_config', default='configs/MVIT_B_32x3_CONV.yaml', type=str, help='config for MViT model')
    parser.add_argument('--mvit_ckpt', default='checkpoints/k600.pyth', type=str, help='checkpoint for MViT model')
    parser.add_argument('--img_pair_model', default='resnet18', type=str, help='resnet18, vit')
    parser.add_argument('--crop_size', type=int, default=2048, help='Size to crop to')
    parser.add_argument('--resize_size', type=int, default=750, help='Size to resize to after cropping')

    # misc
    parser.add_argument('--batch_size', default=64, type=int, help='Batch Size')
    parser.add_argument('--accumulate_grad_batches', default=1, type=int, help='gradient accumulation steps')
    parser.add_argument('--n_threads', default=4, type=int, help='Number of threads for multi-thread loading')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate')
    parser.add_argument('--weight_decay', default=0.0, type=float, help='weight decay')
    parser.add_argument('--lr_step_size', default=20, type=int, help='step size for lr scheduler')
    parser.add_argument('--lr_scheduler', default='cosine', type=str, help='lr scheduler')
    parser.add_argument('--checkpoint', default='', type=str, help='load trained weights')
    parser.add_argument('--partial_ckpt', default='', type=str, help='load partially trained weights')

    parser.add_argument('--log_dir', default='', type=str, help='directory to store logs')
    parser.add_argument('--enable_progress_bar', action="store_true", help='enable progress bar')
    parser.add_argument('--max_epochs', default=50, type=int, help='maximum epochs')


    args = parser.parse_args()


    monitor = 'val_loss'
    mode = 'min'

    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        save_last=True,
        verbose=False,
        monitor=monitor,
        every_n_epochs=1,
        mode=mode,
        save_weights_only=True
    )
    trainer_obj = pl.Trainer(accelerator="gpu", devices="1", log_every_n_steps=1, default_root_dir=args.log_dir, max_epochs=args.max_epochs, accumulate_grad_batches=args.accumulate_grad_batches, enable_progress_bar=args.enable_progress_bar, callbacks=[checkpoint_callback])
    model = trainer.FailureClassificationTrainer(args)
    if args.checkpoint != '':
        model = trainer.FailureClassificationTrainer.load_from_checkpoint(args.checkpoint)
        trainer_obj.test(model)
    else:
        trainer_obj.fit(model)
        # load the last saved model
        model = trainer.FailureClassificationTrainer.load_from_checkpoint(checkpoint_callback.last_model_path)
        trainer_obj.test(model)



if __name__ == '__main__':
    main()
