import h5py
import numpy as np
from DPAD import DPADModel
from tqdm import tqdm

# selectedMethodCode = 'DPAD_RTR2_CzCy1HL64U_ErSV16' when spikes were int
# postfix = "Cz" # best for 09-15 (anything) and 10-07 (behavior)
# postfix = "A" # best for 10-07 (forward prediction)

for postfix in ["Cz", "A"]:
    selectedMethodCode = f"DPAD_RTR2_{postfix}1HL64U_ErSV16"

    n_factors = 100
    n_beh_factors = 40
    epochs = 2500  # Default for this is 2500.

    experiments = [
        "Chewie_CO_FF_2016-09-15",
        "Chewie_CO_FF_2016-09-21",
        "Chewie_CO_FF_2016-10-05",
        "Chewie_CO_FF_2016-10-07",
        "Mihili_CO_FF_2014-02-03",  # *
        "Mihili_CO_FF_2014-02-17",  # + (only BL)
        "Mihili_CO_FF_2014-02-18",
        "Mihili_CO_FF_2014-03-07",
    ]

    def r2_score(y_true, y_pred):
        y_true = y_true.reshape(-1, y_true.shape[-1])
        y_pred = y_pred.reshape(-1, y_pred.shape[-1])
        SS_res = np.sum((y_true - y_pred) ** 2, axis=0)
        SS_tot = np.sum((y_true - np.mean(y_true, axis=0)) ** 2, axis=0)
        return np.mean(1 - SS_res / SS_tot)

    def save_results(f, area, train_outputs, test_outputs):
        if f"train_{area}_dpad{n_beh_factors}{postfix}_pred" in f:
            del f[f"train_{area}_dpad{n_beh_factors}{postfix}_pred"]
        if f"test_{area}_dpad{n_beh_factors}{postfix}_pred" in f:
            del f[f"test_{area}_dpad{n_beh_factors}{postfix}_pred"]
        f.create_dataset(
            f"train_{area}_dpad{n_beh_factors}{postfix}_pred", data=train_outputs
        )
        f.create_dataset(
            f"test_{area}_dpad{n_beh_factors}{postfix}_pred", data=test_outputs
        )

    summary_dict = {}
    for short_dataset_name in tqdm(experiments):
        summary_dict[short_dataset_name] = {}

        for area in ["all", "PMd", "M1"]:
            dataset_name = f"{short_dataset_name}_session_vel_{area}_spikes_go"
            loadpath = f"/disk/scratch/nkudryas/BAND-torch/datasets/\
                {dataset_name}.h5"

            data = {}
            with h5py.File(loadpath, "r") as h5file:
                train_data = h5file["train_recon_data"][()].astype(np.int32)
                valid_data = h5file["valid_recon_data"][()].astype(np.int32)
                train_behavior = h5file["train_behavior"][()].astype(np.float32)
                valid_behavior = h5file["valid_behavior"][()].astype(np.float32)
                train_epoch = h5file["train_epoch"][()].astype(np.int32)
                valid_epoch = h5file["valid_epoch"][()].astype(np.int32)
                train_inds = h5file["train_inds"][()].astype(np.int32)
                valid_inds = h5file["valid_inds"][()].astype(np.int32)

                train_target_direction = h5file["train_target_direction"][()].astype(
                    np.float32
                )
                valid_target_direction = h5file["valid_target_direction"][()].astype(
                    np.float32
                )

                for key in h5file.keys():
                    data[key] = h5file[key][()]

            idSysF = DPADModel()
            args = DPADModel.prepare_args(selectedMethodCode)
            yTrain = train_data.reshape((-1, train_data.shape[-1])) / 0.01
            zTrain = train_behavior.reshape((-1, train_behavior.shape[-1]))
            idSysF.fit(
                yTrain.T,
                Z=zTrain.T,
                nx=n_factors,
                n1=n_beh_factors,
                epochs=epochs,
                **args,
            )
            zTrainPredF, _, _ = idSysF.predict(
                yTrain
            )  # Run inference to generate predictions (Train test smoothing first)

            yTest = valid_data.reshape((-1, valid_data.shape[-1])) / 0.01
            zTest = valid_behavior.reshape((-1, valid_behavior.shape[-1]))
            zTestPredF, yTestPredF, xTestPredF = idSysF.predict(
                yTest
            )  # Run inference to generate predictions

            train_outputs = zTrainPredF.reshape(train_behavior.shape)
            test_outputs = zTestPredF.reshape(valid_behavior.shape)

            final_r2 = r2_score(valid_behavior, test_outputs).item()
            summary_dict[short_dataset_name][f"{area}_R2_all"] = np.round(
                100 * final_r2, 1
            )

            for e, epoch_name in enumerate(["BL", "AD", "WO"]):
                final_r2 = r2_score(
                    valid_behavior[valid_epoch == e], test_outputs[valid_epoch == e]
                ).item()
                summary_dict[short_dataset_name][f"{area}_R2_{epoch_name}"] = np.round(
                    100 * final_r2, 1
                )

            # save predictions
            results_path = f"./results/{short_dataset_name}.h5"
            with h5py.File(results_path, "a") as f:
                save_results(f, area, train_outputs, test_outputs)

    # save summary
    with open(f"./results/DPAD{n_beh_factors}{postfix}_decoder_R2.csv", "w") as f:
        get_column_names = summary_dict[experiments[0]].keys()
        f.write("Dataset,")
        for key in get_column_names:
            f.write(f"{key},\t")
        f.write("\n")
        for key in summary_dict.keys():
            f.write(f"{key},\t")
            for key2 in get_column_names:
                f.write(f"{summary_dict[key][key2]},\t")
            f.write("\n")
