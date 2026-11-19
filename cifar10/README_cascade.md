# Cascade Denoised Smoothing — patch cho `ethz-privsec/diffusion_denoised_smoothing`

Ý tưởng: base classifier `f(x+ε)` = đường rẻ (ResNet-110 huấn luyện với nhiễu của Cohen, chạy thẳng trên x+ε);
chỉ khi `max softmax < τ` mới gọi đường đắt (diffusion một bước + ViT của Carlini).
Vì quyết định chỉ phụ thuộc x+ε và τ cố định trước, CERTIFY của Cohen giữ nguyên hiệu lực.

## 0. Cài đặt

```bash
git clone https://github.com/ethz-privsec/diffusion_denoised_smoothing.git
cd diffusion_denoised_smoothing
conda create -n dds python=3.10 -y && conda activate dds
pip install torch torchvision timm transformers statsmodels scipy numpy

# Checkpoint diffusion CIFAR-10 (iDDPM 50M)
wget -O cifar10/cifar10_uncond_50M_500K.pt \
  https://openaipublic.blob.core.windows.net/diffusion/march-2021/cifar10_uncond_50M_500K.pt

# Kiến trúc ResNet-110 của Cohen (cần cho đường rẻ)
git clone https://github.com/locuslab/smoothing.git /tmp/cohen
cp -r /tmp/cohen/code/archs cifar10/archs
# Tải models.tar từ Google Drive trong README của locuslab/smoothing, giải nén ra ./models
# -> models/cifar10/resnet110/noise_0.25/checkpoint.pth.tar (và noise_0.50, noise_1.00)
```

**Bắt buộc sửa `cifar10/core.py`** (SciPy ≥ 1.12 đã bỏ `binom_test`, file gốc sẽ lỗi khi import):

```bash
sed -i 's/from scipy.stats import norm, binom_test/from scipy.stats import norm, binomtest/; s/if binom_test(count1, count1 + count2, p=0.5) > alpha:/if binomtest(count1, count1 + count2, p=0.5).pvalue > alpha:/' cifar10/core.py
```

Chép 5 file của patch vào `cifar10/`: `DRM_cascade.py`, `probe.py`, `calibrate_tau.py`, `certify_cascade.py`, `compare.py`.
Mọi lệnh dưới đây chạy từ thư mục gốc repo (vì đường dẫn checkpoint là `cifar10/...`), với `PYTHONPATH=cifar10`.

```bash
export PYTHONPATH=cifar10
CKPT=models/cifar10/resnet110/noise_0.25/checkpoint.pth.tar
mkdir -p out data
```

## 1. Tái lập baseline gốc (chưa sửa gì)

```bash
python cifar10/certify.py --sigma 0.25 --skip 20 --N0 100 --N 10000 --alpha 0.0005 \
  --batch_size 500 --outfile out/orig_0.25.tsv
python cifar10/compare.py ref=PATH/Dual-Randomized-Smoothing/reproduce/data/carlini/0.25.tsv \
  orig=out/orig_0.25.tsv --skip 20
```
Mốc (500 ảnh, N=10k): r=0 → 86.8, r=0.25 → 73.8, r=0.5 → 55.6, r=0.75 → 36.2, ~14 s/ảnh.
Lệch ≤ 1–2 điểm là tái lập đạt.

## 2. Đo đường rẻ (phép thử khả thi, < 1 giờ GPU)

```bash
python cifar10/probe.py --sigma 0.25 --cheap_ckpt $CKPT --num_images 9500 --K 4 \
  --split test_holdout --certify_skip 20 --out out/probe_0.25.npz
```
In ra: accuracy của đường rẻ / DDS, tỉ lệ đồng thuận, và với mỗi τ: tỉ lệ chấp nhận (= phần gọi diffusion tiết kiệm được)
cùng P(chấp nhận & bất đồng).

Dữ liệu hiệu chỉnh mặc định là 9.500 ảnh **test có idx % 20 ≠ 0**, rời hẳn với 500 ảnh certify (idx % 20 = 0).
Không dùng ảnh train: ResNet của Cohen và ViT đều đã học trên ảnh train, nên trên đó đường rẻ tự tin hơn thật
và rủi ro hiệu chỉnh bị lạc quan, khiến bảo đảm δ không chuyển được sang tập test. (`--split train` vẫn còn để làm ablation.)

## 3. Chọn τ có bảo đảm thống kê

```bash
python cifar10/calibrate_tau.py out/probe_0.25.npz --delta 0.01 --alpha 0.05
```
Fixed-sequence testing + Clopper–Pearson: với xác suất ≥ 1−α, P(chấp nhận & bất đồng với DDS) ≤ δ.

## 4. Certify ba chế độ trên cùng 500 ảnh test

```bash
for MODE in dds cheap cascade; do
  python cifar10/certify_cascade.py --mode $MODE --tau <TAU_TỪ_BƯỚC_3> --sigma 0.25 \
    --cheap_ckpt $CKPT --skip 20 --N 10000 --alpha 0.0005 --outfile out/${MODE}_0.25.tsv
done
python cifar10/compare.py dds=out/dds_0.25.tsv cheap=out/cheap_0.25.tsv cascade=out/cascade_0.25.tsv
```
Cột `exp_frac` = tỉ lệ mẫu nhiễu phải đi đường đắt; `sec/img` = thời gian thật.

Lưu ý: `--mode dds` thêm nhiễu đúng σ trong không gian [0,1] rồi scale theo √ᾱ_t (như công thức trong paper Carlini).
`certify.py` gốc thì dùng `q_sample` tại t ⇒ nhiễu thực tế hơi lớn hơn σ. Hai bản lệch nhau rất ít;
so cascade với `dds` của patch để công bằng tuyệt đối, và dùng bước 1 chỉ để kiểm tra tái lập.

## Chạy trên Slurm (thay cho các bước 0–4)

```bash
bash slurm/setup.sh                    # login node, 1 lần: env, checkpoint, CIFAR-10, cache ViT, sanity check
export SBATCH_PARTITION=<gpu-partition> SBATCH_ACCOUNT=<account>   # nếu cụm yêu cầu
bash slurm/submit_all.sh 0.25          # orig, dds, cheap, probe(+calibrate), cascade(τ hiệu chỉnh), sweep τ
squeue -u $USER
bash slurm/report.sh 0.25              # bảng certified acc / sec/img / exp_frac
```
Biến môi trường tùy chỉnh: `N`, `SKIP`, `BATCH` (giảm nếu OOM), `DELTA`, `SWEEP="0.5 0.7 0.9 0.99"`, `CONDA_ENV`.
Kết quả: `out/{orig,dds,cheap}_<σ>.tsv`, `out/cascade_<σ>_tau{cal,0.9,...}.tsv`, `out/tau_<σ>.txt`.

## 5. Sau CIFAR-10 σ=0.25
- Lặp lại với σ = 0.5, 1.0 (đổi `$CKPT` sang noise_0.50 / noise_1.00).
- Ablation: `--multistep` cho đường đắt; đổi đường rẻ (ResNet-20/56, model ℓ2-robust từ RobustBench).
- Vẽ Pareto certified accuracy vs GPU-giây theo τ.
- ImageNet: viết `imagenet/DRM_cascade.py` tương tự (ResNet-50 của Cohen làm đường rẻ, guided diffusion + BEiT làm đường đắt).
