# Project Name

## Dependencies

To set up the environment, run the following commands:

```bash
conda env create -f ./tools/environment.yml
conda activate rota12
```

## LoFTR and rel_pose

The folders `rel_pose`, `LoFTR` and `pretrained` should be placed in the root directory. You can download these folders from the following URLs:

- [PointVit](https://drive.google.com/drive/folders/1qE4lL2YidXfE6RZXjioyc2IM6TlTUeiM?usp=sharing)
- [LoFTR](https://drive.google.com/drive/folders/1nadVMbujPFvk1Mu_WGMTeUE2MxEMzWTe?usp=sharing)
- [pretrained](https://drive.google.com/drive/folders/1xA6O-FYAKWj0Ed2E3qIu-tKnw29C9q1Z?usp=sharing)

## Evaluation metric files

After running `test.py`, two files are created under `mid_run/<BASE_LINE_NAME>/outputs/` folder:
- `res_all.txt`: Contains all the error metrics.
- `debug_outrot.pkl`: Contains data that allows resume the test evaluation from the last batch that was calculated. If you want to recalculate the test, the `debug_outrot.pkl` file must be deleted.

## Running the baseline test

You can reproduce the `res_all.txt` file, containing the evaluation metrics, using the following command:

```bash
python test.py --config configs/LoFTR/config_cambridge.yaml
```

Example with pre-trained model:

```bash
python test.py --config configs/Extreme_Rotation/config_cambridge.yaml --is_resume False --pretrained pretrained/streetlearnT_cv_distribution.pt
```

## Config files

For each baseline and dataset, there exists a config file under the `configs` folder.
