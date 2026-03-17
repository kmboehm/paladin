import pickle  # nosec

import pandas as pd
import torch
from torch.utils.data import DataLoader

from paladin.data.inference_dataset import InferenceDataset


def main():
    # settings
    histology = "breast"
    rr_df_path = f"../data-commons/freeze/clinical/rr_{histology}.parquet"
    name_to_int_mapping = {
        "AASTR": 0,
        "ACC": 1,
        "ACRM": 2,
        "ACYC": 3,
        "ADNOS": 4,
        "ALUCA": 5,
        "AMPCA": 6,
        "ANGS": 7,
        "ANSC": 8,
        "AODG": 9,
        "APAD": 10,
        "ARMM": 11,
        "ARMS": 12,
        "ASTR": 13,
        "ATM": 14,
        "BA": 15,
        "BCC": 16,
        "BLAD": 17,
        "BLCA": 18,
        "BMGCT": 19,
        "BRCA": 20,
        "BRCANOS": 21,
        "BRCNOS": 22,
        "CCOV": 23,
        "CCRCC": 24,
        "CESC": 25,
        "CHDM": 26,
        "CHOL": 27,
        "CHRCC": 28,
        "CHS": 29,
        "COAD": 30,
        "COADREAD": 31,
        "CSCC": 32,
        "CSCLC": 33,
        "CUP": 34,
        "CUPNOS": 35,
        "DA": 36,
        "DASTR": 37,
        "DDLS": 38,
        "DES": 39,
        "DIFG": 40,
        "DSRCT": 41,
        "DSTAD": 42,
        "ECAD": 43,
        "EGC": 44,
        "EHAE": 45,
        "EHCH": 46,
        "EMPD": 47,
        "EOV": 48,
        "EPDCA": 49,
        "EPIS": 50,
        "EPM": 51,
        "ERMS": 52,
        "ES": 53,
        "ESCA": 54,
        "ESCC": 55,
        "GB": 56,
        "GBAD": 57,
        "GBC": 58,
        "GBM": 59,
        "GCCAP": 60,
        "GEJ": 61,
        "GINET": 62,
        "GIST": 63,
        "GNOS": 64,
        "GRCT": 65,
        "HCC": 66,
        "HGGNOS": 67,
        "HGNEC": 68,
        "HGSFT": 69,
        "HGSOC": 70,
        "HNMUCM": 71,
        "HNSC": 72,
        "IDC": 73,
        "IHCH": 74,
        "ILC": 75,
        "LGGNOS": 76,
        "LGSOC": 77,
        "LMS": 78,
        "LNET": 79,
        "LUAD": 80,
        "LUAS": 81,
        "LUCA": 82,
        "LUNE": 83,
        "LUPC": 84,
        "LUSC": 85,
        "LXSC": 86,
        "MAAP": 87,
        "MACR": 88,
        "MBC": 89,
        "MCC": 90,
        "MDLC": 91,
        "MEL": 92,
        "MFH": 93,
        "MFS": 94,
        "MGCT": 95,
        "MNG": 96,
        "MOV": 97,
        "MPNST": 98,
        "MRLS": 99,
        "MUP": 100,
        "MXOV": 101,
        "NBL": 102,
        "NECNOS": 103,
        "NETNOS": 104,
        "NOT": 105,
        "NPC": 106,
        "NSCLC": 107,
        "NSCLCPD": 108,
        "NSGCT": 109,
        "OCS": 110,
        "OCSC": 111,
        "ODG": 112,
        "OOVC": 113,
        "OPHSC": 114,
        "OS": 115,
        "PAAC": 116,
        "PAAD": 117,
        "PAASC": 118,
        "PAMPCA": 119,
        "PANET": 120,
        "PAST": 121,
        "PDC": 122,
        "PECOMA": 123,
        "PEMESO": 124,
        "PHC": 125,
        "PLBMESO": 126,
        "PLEMESO": 127,
        "PLMESO": 128,
        "PRAD": 129,
        "PRCC": 130,
        "PSEC": 131,
        "PTAD": 132,
        "RBL": 133,
        "RCC": 134,
        "RCSNOS": 135,
        "READ": 136,
        "RMS": 137,
        "SARCNOS": 138,
        "SBC": 139,
        "SBOV": 140,
        "SBWDNET": 141,
        "SCBC": 142,
        "SCCNOS": 143,
        "SCHW": 144,
        "SCLC": 145,
        "SCUP": 146,
        "SDCA": 147,
        "SEM": 148,
        "SFT": 149,
        "SKCM": 150,
        "SOC": 151,
        "SPDAC": 152,
        "SSRCC": 153,
        "STAD": 154,
        "SYNS": 155,
        "TAC": 156,
        "THAP": 157,
        "THHC": 158,
        "THME": 159,
        "THPA": 160,
        "THPD": 161,
        "THYC": 162,
        "THYM": 163,
        "TYST": 164,
        "UCCC": 165,
        "UCEC": 166,
        "UCP": 167,
        "UCS": 168,
        "UCU": 169,
        "UDMN": 170,
        "UEC": 171,
        "ULMS": 172,
        "UM": 173,
        "UMEC": 174,
        "URCC": 175,
        "USARC": 176,
        "USC": 177,
        "UTUC": 178,
        "VMM": 179,
        "VSC": 180,
        "WDLS": 181,
        "WT": 182,
    }
    named_cols_to_drop = [
        "UDMN",
        "ADNOS",
        "CUP",
        "CUPNOS",
        "BRCNOS",
        "GNOS",
        "SCCNOS",
        "PDC",
        "NSCLC",
        "BRCA",
        "SARCNOS",
        "NETNOS",
        "MEL",
        "RCC",
        "BRCANOS",
        "COADREAD",
        "MUP",
        "NECNOS",
        "UCEC",
        "NOT",
    ]
    batch_size = 8
    num_workers = 8
    ckpt_path = "storage/aeon/095kqxr6/checkpoints/095kqxr6.epoch=15-step=8192.model.pkl"  # bahamut
    # ckpt_path = "storage/aeon/0b0dww08/checkpoints/0b0dww08.epoch=15-step=2752.model.pkl"  # yojimbo
    output_df_path = f"../data-commons/freeze/clinical/rr_{histology}_predictions.csv"
    output_part_representations_path = f"../data-commons/freeze/clinical/rr_{histology}_part_representations.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_parquet(rr_df_path)

    col_indices_to_drop = [name_to_int_mapping[x] for x in named_cols_to_drop]
    int_to_name_mapping = dict((v, k) for k, v in name_to_int_mapping.items())

    with open(ckpt_path, "rb") as f:
        model = pickle.load(f)  # nosec
    model.to(device)
    model.eval()

    sample_ids = df["MRN"].tolist() # IMPACT sample id
    sites = df["site"].tolist() # 'Primary' or 'Metastasis'
    tile_tensor_urls = df["optimus_features_tensor_path"].tolist()
    dataset = InferenceDataset(sample_ids=sample_ids, sites=sites, tile_tensor_urls=tile_tensor_urls)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    results_df = []
    part_representations = []

    for batch in dataloader:
        with torch.no_grad():
            batch["tile_tensor"] = batch["tile_tensor"].to(device)
            y = model(batch)
            y["logits"][:, col_indices_to_drop] = -1e6
            batch_size = y["logits"].shape[0]
            for i in range(batch_size):
                softmax = torch.nn.functional.softmax(y["logits"][i], dim=0)
                argmax = torch.argmax(softmax, dim=0)
                class_assignment = int_to_name_mapping[argmax.item()]
                confidence = softmax[argmax].item()
                mean_confidence = torch.mean(softmax).item()
                print(class_assignment, confidence, mean_confidence)
                results_df.append(
                    {
                        "sample_id": batch["sample_id"][i],
                        "class_assignment": class_assignment,
                        "confidence": confidence,
                        "mean_confidence": mean_confidence,
                    }
                )
                part_representations.append(y["whole_part_representation"][i].cpu())

    results_df = pd.DataFrame(results_df)
    results_df.to_csv(output_df_path, index=False)
    part_representations = torch.stack(part_representations, dim=0)
    torch.save(part_representations, output_part_representations_path)


if __name__ == "__main__":
    main()
