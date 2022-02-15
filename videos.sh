#ffmpeg -i results/inference/gaussian_tanh/l1/unnorm/gt/img%03d.png -pix_fmt yuv420p results/inference/videos/gaussian_tanh/ll1_norm1_gt.mp4
#ffmpeg -i results/inference/gaussian_tanh/l1/unnorm/rec/img%03d.png -pix_fmt yuv420p results/inference/videos/gaussian_tanh/un_l1_norm1_rec.mp4
#ffmpeg -i results/inference/gaussian_tanh/mse/unnorm/gt/img%03d.png -pix_fmt yuv420p results/inference/videos/gaussian_tanh/lmse_norm1_gt.mp4
#ffmpeg -i results/inference/gaussian_tanh/mse/unnorm/rec/img%03d.png -pix_fmt yuv420p results/inference/videos/gaussian_tanh/basun_mse_norm1_rec.mp4
# ffmpeg -i results/inference/gaussian_small/l1/norm/rec/img%03d.png -pix_fmt yuv420p results/inference/videos/gaussian_small/l1_norm_rec.mp4
# ffmpeg -i results/inference/gaussian_small/l1/norm1/gt/img%03d.png -pix_fmt yuv420p results/inference/videos/gaussian_small/l1_norm1_gt.mp4
# ffmpeg -i results/inference/gaussian_small/l1/norm1/rec/img%03d.png -pix_fmt yuv420p results/inference/videos/gaussian_small/l1_norm1_rec.mp4
# ffmpeg -i results/inference/gaussian_small/l1/unnorm/gt/img%03d.png -pix_fmt yuv420p results/inference/videos/gaussian_small/l1_unnorm_gt.mp4
# ffmpeg -i results/inference/gaussian_small/l1/unnorm/rec/img%03d.png -pix_fmt yuv420p results/inference/videos/gaussian_small/l1_unnorm_rec.mp4
# ffmpeg -i results/inference/gaussian_small/mse/norm/gt/img%03d.png -pix_fmt yuv420p results/inference/videos/gaussian_small/mse_norm_gt.mp4
# ffmpeg -i results/inference/gaussian_small/mse/norm/rec/img%03d.png -pix_fmt yuv420p results/inference/videos/gaussian_small/mse_norm_rec.mp4
# ffmpeg -i results/inference/gaussian_small/mse/norm1/gt/img%03d.png -pix_fmt yuv420p results/inference/videos/gaussian_small/mse_norm1_gt.mp4
# ffmpeg -i results/inference/gaussian_small/mse/norm1/rec/img%03d.png -pix_fmt yuv420p results/inference/videos/gaussian_small/mse_norm1_rec.mp4
# ffmpeg -i results/inference/gaussian_small/mse/unnorm/gt/img%03d.png -pix_fmt yuv420p results/inference/videos/gaussian_small/mse_unnorm_gt.mp4


# ffmpeg -i results/inference/gaussian/wmse/unnorm/rec/img%03d.png -pix_fmt yuv420p results/inference/videos/gaussian_tanh/ll1_norm1_gt.mp4
# ffmpeg -i results/inference/gaussian_small/wmse/unnorm/rec/img%03d.png -pix_fmt yuv420p results/inference/videos/gaussian_tanh/ll1_norm1_gt.mp4


ffmpeg -i results/inference/gaussian/wmse/unnorm/rec/img%03d.png results/inference/videos/gaussian/wmse_unnorm_rec.gif 
ffmpeg -i results/inference/gaussian_small/wmse/unnorm/rec/img%03d.png results/inference/videos/gaussian_small/wmse_unnorm_rec.gif
