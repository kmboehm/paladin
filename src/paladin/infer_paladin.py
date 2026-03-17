import pickle  # nosec

import pandas as pd
import torch
from torch.utils.data import DataLoader

from paladin.data.inference_dataset import InferenceDataset


def logits_to_point_estimates(logits):
    # logits is a tensor of shape (batch_size, 2 * (n_clf_tasks + n_reg_tasks))
    # need to convert it to a tensor of shape (batch_size, n_clf_tasks + n_reg_tasks)
    return logits[:, ::2] / (logits[:, ::2] + logits[:, 1::2])


def main():
    # settings
    histology = "rectum"
    rr_df_path = f"../data-commons/freeze/applications/rr_{histology}.parquet"
    batch_size = 16
    num_workers = 16
    ckpt_path = "storage/paladin/ef9g2g05/checkpoints/ef9g2g05.epoch=19-step=80.model.pkl"
    output_df_path = f"../data-commons/freeze/clinical/rr_{histology}_predictions.csv"
    output_part_representations_path = f"../data-commons/freeze/clinical/rr_{histology}_part_representations.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_parquet(rr_df_path)
    print(df)
    print(df.columns.tolist())

    with open(ckpt_path, "rb") as f:
        model = pickle.load(f)  # nosec
    model.to(device)
    model.eval()

    sample_ids = df["Sample_ID"].tolist()
    tile_tensor_urls = df["optimus_features_tensor_path"].tolist()
    dataset = InferenceDataset(sample_ids=sample_ids, tile_tensor_urls=tile_tensor_urls)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    results_df = []
    part_representations = []

    for batch in dataloader:
        with torch.no_grad():
            batch["tile_tensor"] = batch["tile_tensor"].to(device)
            outputs = model(batch)
            logits = outputs["logits"]
            # Apply softplus to ensure positive values for beta-binomial parameters
            logits = torch.nn.functional.softplus(logits) + 1.0  # to enforce concavity
            point_estimates = logits_to_point_estimates(logits)
            batch_size = point_estimates.shape[0]
            for i in range(batch_size):
                class_assignment = point_estimates[i].item()
                print(class_assignment)
                results_df.append({"sample_id": batch["sample_id"][i], "class_assignment": class_assignment})
                part_representations.append(outputs["whole_part_representation"][i])

    results_df = pd.DataFrame(results_df)
    results_df.to_csv(output_df_path, index=False)
    part_representations = torch.cat(part_representations, dim=0)
    torch.save(part_representations, output_part_representations_path)


if __name__ == "__main__":
    main()
