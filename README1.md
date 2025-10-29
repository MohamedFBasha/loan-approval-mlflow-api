conda create -n mlflow_env python=3.10.9
conda activate mlflow_env
pip install -r requirements.txt
mlflow ui


conda activate mlflow_env
pip install uvicorn fastapi
uvicorn main:app --host 127.0.0.1 --port 8000 --reload

``` python
# exmaple of data to be sent
{
    "columns": [
        "no_of_dependents",
        "education",
        "self_employed",
        "income_annum",
        "loan_amount",
        "loan_term",
        "cibil_score",
        "residential_assets_value",
        "commercial_assets_value",
        "luxury_assets_value",
        "bank_asset_value"
    ],
    "data": [
        [-0.2914039977413991, 1.0040695112580063, -1.0040695112580063, 0.37255141623584065, 0.9487081783858019, 0.5456702102315741, -0.5218397554059292, 0.2450565069160716, 1.0302355939926513, -0.30197786093753753, -0.49064925537831916],
        [-1.4721434873239536, -0.995946982542167, 0.995946982542167, 1.1232492205160998, 1.6450682129252223, -0.8455606483516742, -1.129646438050548, -0.5482513082409929, -0.9573447306643038, 0.960558969153455, 0.22141599454712899]
    ]
}
#........................................................................................#