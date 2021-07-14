
import pathlib

data_path = pathlib.Path(__file__).resolve().parent

dundee_data_path = data_path.joinpath("dundee")
dundee_data_path_V2 = data_path.joinpath("dundee_V2")

cerm_data_path = data_path.joinpath("Annotated_spectra")
cerm_csv_data_path = data_path.joinpath("cerm_csv")