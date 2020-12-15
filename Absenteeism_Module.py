{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "Absenteeism Module",
      "provenance": [],
      "mount_file_id": "1UpvVb0KyjvTb4xFFnU1qgxdXHe1FOBpC",
      "authorship_tag": "ABX9TyPLUnkoFYhafFmzQDBB+bSi",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/Jacquesjh/Absenteeism/blob/main/Absenteeism_Module.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "jqZpWg8Ga5Lt"
      },
      "source": [
        "#Iremos criar um modulo para utilizarmos nossos códigos futuramente, não haverão comentários para simplificar código"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "YalOBlP8bLwq"
      },
      "source": [
        "import pandas as pd\r\n",
        "import numpy as np\r\n",
        "import pickle\r\n",
        "from sklearn import preprocessing\r\n",
        "from sklearn import model_selection\r\n",
        "from sklearn import metrics\r\n",
        "from sklearn.linear_model import LogisticRegressionCV\r\n",
        "\r\n",
        "class absenteeism_module():\r\n",
        "\r\n",
        "  def __init__(self, model_file, scaler_file):\r\n",
        "    #abre o modelo e escalador salvos\r\n",
        "    with open('Logistic Regression Model', 'wb') as model_file, open('Absenteeism Scaler', 'wb') as scaler_file:\r\n",
        "      self.reg = pickle.load(model_file)\r\n",
        "      self.scaler = pickle.load(scaler_file)\r\n",
        "      self.data = None\r\n",
        "\r\n",
        "  #Para pré processar os dados de um .csv\r\n",
        "  def load_and_clean_data(self, data_file):\r\n",
        "\r\n",
        "    df = pd.read_csv(data_file, delimiter = ',')\r\n",
        "    self.df_with_predictions = df.copy()\r\n",
        "\r\n",
        "    df = df.drop(['ID'], axis = 1)\r\n",
        "    df['Absenteeism Time in Hours'] = 'NaN'\r\n",
        "\r\n",
        "    reasons_columns = pd.get_dummies(df.['Reasons for Absence'], drop_first = True)\r\n",
        "\r\n",
        "    reason_type1 = reasons_columns.iloc[:, 1:14].max(axis = 1)\r\n",
        "    reason_type2 = reasons_columns.iloc[:, 15:17].max(axis = 1)\r\n",
        "    reason_type3 = reasons_columns.iloc[:, 18:21].max(axis = 1)\r\n",
        "    reason_type4 = reasons_columns.iloc[:, 22:].max(axis = 1)\r\n",
        "\r\n",
        "    df = df.drop(['Reasons for Absence'], axis = 1)\r\n",
        "\r\n",
        "    df. pd.concat([df, reason_type1, reason_type2, reason_type3, reason_type4], axis = 1)\r\n",
        "\r\n",
        "    column_names = ['Date', 'Transportation Expense', 'Distance to Work', 'Age',\r\n",
        "       'Daily Work Load Average', 'Body Mass Index', 'Education',\r\n",
        "       'Children', 'Pets', 'Absenteeism Time in Hours', 'Reason 1', 'Reason 2', 'Reason 3', 'Reason 4']\r\n",
        "\r\n",
        "    df.columns = column_names\r\n",
        "\r\n",
        "    column_names_ordered = ['Reason 1', 'Reason 2', 'Reason 3', 'Reason 4', 'Date', 'Transportation Expense', 'Distance to Work', 'Age',\r\n",
        "       'Daily Work Load Average', 'Body Mass Index', 'Education',\r\n",
        "       'Children', 'Pets', 'Absenteeism Time in Hours']\r\n",
        "    \r\n",
        "    df = df[column_names_ordered]\r\n",
        "\r\n",
        "    df['Date'] = pd.to_datetime(df['Date'], format = '%d/%m/%Y')\r\n",
        "\r\n",
        "    list_months = []\r\n",
        "    list_weekday = []\r\n",
        "    for i in range(df['Date'].shape[0]):\r\n",
        "      list_months.append(df['Date'][i].month)\r\n",
        "      list_weekday.append(df['Date'][i].weekday)\r\n",
        "\r\n",
        "    df['Month Value'] = list_months\r\n",
        "    df['Day of the Week'] = list_weekday\r\n",
        "\r\n",
        "    df.drop(['Date'], axis = 1)\r\n",
        "\r\n",
        "    new_column_names = ['Reason 1', 'Reason 2', 'Reason 3', 'Reason 4', 'Day of the Week', 'Month Value', 'Transportation Expense', 'Distance to Work', 'Age',\r\n",
        "       'Daily Work Load Average', 'Body Mass Index', 'Education',\r\n",
        "       'Children', 'Pets', 'Absenteeism Time in Hours']\r\n",
        "\r\n",
        "    df['Education'] = df['Education'].map({1:0, 2:1, 3:1, 4:1})\r\n",
        "\r\n",
        "    df = df.fillna(value = 0)\r\n",
        "\r\n",
        "    df = df.drop(['Absenteeism Time in Hours'], axis = 1)\r\n",
        "\r\n",
        "    self.preprocessed_data = df.copy()\r\n",
        "\r\n",
        "    self.data = self.scaler.transform(df)\r\n",
        "\r\n",
        "  def predicted_prob(self):\r\n",
        "    if(self.data is not None):\r\n",
        "      pred = self.reg.predict_proba(self.data)\r\n",
        "      return pred\r\n",
        "\r\n",
        "  def predict_target_category(self):\r\n",
        "    if (self.data is not None):\r\n",
        "      pred_targets = self.reg.predict(sel.data)\r\n",
        "      return pred_targets\r\n",
        "\r\n",
        "  def predict_targets(self):\r\n",
        "    if(self.data is not None):\r\n",
        "      self.preprocessed_data['Probability'] = self.reg.predict_proba(self.data)[:, 1]\r\n",
        "      self.preprocessed_data['Prediction'] = self.reg.predict(self.data)\r\n",
        "      return self.preprocessed_data\r\n"
      ],
      "execution_count": null,
      "outputs": []
    }
  ]
}