#Infer reconstruction with recurrency enabled
# python reconstruct.py -m model/lpips/F_LPIPS_24.pth -o results/inference/lpips_f/
# ffmpeg -i results/inference/lpips_f/rec/img%03d.png results/inference/videos/f_lpips_rec.gif 
# ffmpeg -i results/inference/lpips_f/gt/img%03d.png results/inference/videos/f_lpips_gt.gif 

# python reconstruct.py -m model/lpips/F_LPIPS_MSE_30.pth -o results/inference/lpips_mse_f/
# ffmpeg -i results/inference/lpips_mse_f/gt/img%03d.png results/inference/videos/f_lpips_mse_gt.gif 
# ffmpeg -i results/inference/lpips_mse_f/rec/img%03d.png results/inference/videos/f_lpips_mse_rec.gif 

#Infer reconstruction with recurrency disabled
#python reconstruct_nornn.py -m model/lpips/TR_4_LPIPS_F.pth -o results/inference/lpips_rec_full/

#Infer reconstruction with recurrency enabled sequences
# python reconstruct_seq.py -m model/seq/M_SEQ_4_20.pth -o results/inference/M_SEQ_4_20/ -i data/gaussian_sequences_single/train/
# python reconstruct_seq.py -m model/seq/M_SEQ_4.pth -o results/inference/M_SEQ_4/ -i data/gaussian_sequences_single/train/
# python reconstruct_seq.py -m model/seq/M_SEQ_5_55.pth -o results/inference/M_SEQ_5_55/ -i data/gaussian_sequences_single/train/
# python reconstruct_seq.py -m model/seq/M_SEQ_5.pth -o results/inference/M_SEQ_5/ -i data/gaussian_sequences_single/train/
# ffmpeg -i results/inference/M_SEQ_4_20/rec/img%03d.png results/inference/videos/M_SEQ_4_20.gif 
# ffmpeg -i results/inference/M_SEQ_4/rec/img%03d.png results/inference/videos/M_SEQ_4.gif 
# ffmpeg -i results/inference/M_SEQ_5_55/rec/img%03d.png results/inference/videos/M_SEQ_5_55.gif 
# ffmpeg -i results/inference/M_SEQ_5/rec/img%03d.png results/inference/videos/M_SEQ_5.gif 

# python reconstruct.py -m model/lpips/F_LPIPS_MSE_30.pth -o results/inference/lpips_mse_f/
# ffmpeg -i results/inference/lpips_mse_f/gt/img%03d.png results/inference/videos/f_lpips_mse_gt.gif 
# ffmpeg -i results/inference/lpips_mse_f/rec/img%03d.png results/inference/videos/f_lpips_mse_rec.gif 

#Infer reconstruction with recurrency enabled sequences - trained E2VID
#python reconstruct_e2vid.py -m model/e2vid_trained/cnn_e2vid_lpips_m.pth -em model/e2vid_trained/rnn_e2vid_lpips_m.pth -o results/inference/e2vid_lpips/ 
#python reconstruct_e2vid.py -m model/e2vid_trained/cnn_e2vid_lpips_m_49.pth -em model/e2vid_trained/rnn_e2vid_lpips_m_49.pth -o results/inference/e2vid_lpips_49/
#python reconstruct_e2vid.py -m model/e2vid_trained/cnn_e2vid_mselpips_m.pth -em model/e2vid_trained/rnn_e2vid_mselpips_m.pth -o results/inference/e2vid_mselpips/ 
#python reconstruct_e2vid.py -m model/e2vid_trained/cnn_e2vid_mselpips_m_64.pth -em model/e2vid_trained/rnn_e2vid_mselpips_m_64.pth -o results/inference/e2vid_mselpips_64/
# python reconstruct_only.py -em model/e2vid_trained/rnn_e2vid_only_m.pth -o results/inference/e2vid_only/ 
#python reconstruct_only.py -em model/e2vid_trained/rnn_e2vid_only_m_61.pth -o results/inference/e2vid_only_61/ -i data/gaussian_sequences_single/test/

# ffmpeg -i results/inference/e2vid_lpips/rec/img%03d.png results/inference/videos/e2vid_lpips.gif 
# ffmpeg -i results/inference/e2vid_lpips_49/rec/img%03d.png results/inference/videos/e2vid_lpips_49.gif
# ffmpeg -i results/inference/e2vid_mselpips/rec/img%03d.png results/inference/videos/e2vid_mselpips.gif 
# ffmpeg -i results/inference/e2vid_mselpips_64/rec/img%03d.png results/inference/videos/e2vid_mselpips_64.gif
# ffmpeg -i results/inference/e2vid_only/rec/img%03d.png results/inference/videos/e2vid_only.gif 
#ffmpeg -i results/inference/e2vid_only_61/rec/img%03d.png results/inference/videos/e2vid_only_61.gif
#ffmpeg -i results/inference/e2vid_only_61/gt/img%03d.png results/inference/videos/e2vid_only_61_gt.gif 

#Reconstruct with frozen E2VID only
#python reconstruct_only_frozen.py -o results/inference/e2vid_blurred/ -i data/gaussian_sequences_single/single/
#ffmpeg -i results/inference/e2vid_blurred/rec/img%03d.png results/inference/videos/e2vid_blurred.gif 

#Infer reconstruction with recurrency enabled sequences - trained E2VID
python reconstruct_e2vid.py -m model/e2vid_trained/cnn_e2vid_mselpips_f_5.pth -em model/e2vid_trained/rnn_e2vid_mselpips_f_5.pth -o results/inference/e2vid_mselpips_f/ -i data/gaussian_sequences_single/Dog1/
ffmpeg -i results/inference/e2vid_mselpips_f/rec/img%03d.png results/inference/videos/e2vid_mselpips_f_Dog1.gif
