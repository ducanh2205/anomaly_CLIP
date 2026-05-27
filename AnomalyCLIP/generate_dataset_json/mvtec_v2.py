import os
import json


class MVTecAD2Solver(object):
    CLSNAMES = [
        'can', 'fabric', 'fruit_jelly', 'rice',
        'sheet_metal', 'vial', 'wallplugs', 'walnuts',
    ]

    def __init__(self, root='data/mvtec_ad2'):
        self.root = root
        self.meta_path = f'{root}/meta.json'

    def run(self):
        info = dict(train={}, test={})
        anomaly_samples = 0
        normal_samples = 0

        for cls_name in self.CLSNAMES:
            cls_dir = f'{self.root}/{cls_name}'

            # ── TRAIN ─────────────────────────────────────────────────────
            train_info = []
            train_good_dir = f'{cls_dir}/train/good'
            if os.path.exists(train_good_dir):
                for img_name in sorted(os.listdir(train_good_dir)):
                    if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        continue
                    train_info.append(dict(
                        img_path=f'{cls_name}/train/good/{img_name}',
                        mask_path='',
                        cls_name=cls_name,
                        specie_name='good',
                        anomaly=0,
                    ))
            info['train'][cls_name] = train_info

            # ── TEST ──────────────────────────────────────────────────────
            test_info = []

            # good images
            test_good_dir = f'{cls_dir}/test_public/good'
            if os.path.exists(test_good_dir):
                for img_name in sorted(os.listdir(test_good_dir)):
                    if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        continue
                    test_info.append(dict(
                        img_path=f'{cls_name}/test_public/good/{img_name}',
                        mask_path='',
                        cls_name=cls_name,
                        specie_name='good',
                        anomaly=0,
                    ))
                    normal_samples += 1

            # bad images
            # mask: ground_truth/bad/{stem}_mask.png
            test_bad_dir = f'{cls_dir}/test_public/bad'
            gt_dir       = f'{cls_dir}/test_public/ground_truth/bad'
            if os.path.exists(test_bad_dir):
                for img_name in sorted(os.listdir(test_bad_dir)):
                    if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        continue
                    stem = os.path.splitext(img_name)[0]          # e.g. 000_overexposed
                    mask_name = f'{stem}_mask.png'                 # e.g. 000_overexposed_mask.png
                    mask_full = os.path.join(gt_dir, mask_name)

                    mask_path = (
                        f'{cls_name}/test_public/ground_truth/bad/{mask_name}'
                        if os.path.exists(mask_full) else ''
                    )

                    test_info.append(dict(
                        img_path=f'{cls_name}/test_public/bad/{img_name}',
                        mask_path=mask_path,
                        cls_name=cls_name,
                        specie_name='bad',
                        anomaly=1,
                    ))
                    anomaly_samples += 1

            info['test'][cls_name] = test_info

        with open(self.meta_path, 'w') as f:
            f.write(json.dumps(info, indent=4) + '\n')

        print(f'Saved: {self.meta_path}')
        print(f'normal_samples: {normal_samples}  anomaly_samples: {anomaly_samples}')


if __name__ == '__main__':
    runner = MVTecAD2Solver(
        root=r'C:\\anomaly_detection\\data\\mvtec_ad_2'
    )
    runner.run()
