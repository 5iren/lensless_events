python reconstruct.py -m "model/wmse/arch(UNORM_WMSE)-e(200)-l(MSE)-o(Adam)-lr(0.001).pth" -o results/inference/gaussian_small/wmse/ -d unnorm -i data/single_gaussian_small/
python reconstruct.py -m "model/wmse/arch(UNORM_WMSE)-e(201)-l(MSE)-o(Adam)-lr(0.001).pth" -o results/inference/gaussian/wmse/ -d unnorm -i data/single_gaussian/
#python reconstruct.py -m "model/gaussian_tanh/arch(UNORM)-e(200)-l(L1)-o(Adam)-lr(0.0001).pth" -o results/inference/gaussian_small/l1/ -d unnorm
#python reconstruct.py -m "model/gaussian_tanh/arch(NORM)-e(200)-l(MSE)-o(Adam)-lr(0.0001).pth" -o results/inference/gaussian_small/mse/ -d norm
#python reconstruct.py -m "model/gaussian_tanh/arch(NORM1)-e(200)-l(MSE)-o(Adam)-lr(0.0001).pth" -o results/inference/gaussian_small/mse/ -d norm1
#python reconstruct.py -m "model/gaussian_tanh/arch(UNORM)-e(200)-l(MSE)-o(Adam)-lr(0.0001).pth" -o results/inference/gaussian_small/mse/ -d unnorm
