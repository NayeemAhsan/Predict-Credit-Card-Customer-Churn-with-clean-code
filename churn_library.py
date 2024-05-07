# library doc string
"""
Helper functions to Predict Credit Card Customer Chrun
Author: Nayeem Ahsan
Date: 4/18/2024
"""

# import libraries
import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import seaborn as sns
sns.set()
os.environ['QT_QPA_PLATFORM'] = 'offscreen'


def import_data(pth):
    '''
    Load data from a CSV file.

    Parameters:
        pth (str): Path to the CSV file.

    Returns:
        DataFrame: Pandas DataFrame containing the data.
    '''
    dataframe = pd.read_csv(pth, index_col=0)

    return dataframe


def data_cleaning(dataframe):
    '''
    Perform data cleaning on the DataFrame.

    Parameters:
        dataframe (DataFrame): Pandas DataFrame to clean.

    Returns:
        DataFrame: Cleaned DataFrame.
    '''
    cleaned_df = dataframe.copy()
    # Encode Churn dependent variable : 0 = Did not churned ; 1 = Churned
    cleaned_df['Churn'] = np.where(
        cleaned_df['Attrition_Flag'] == "Existing Customer", 0, 1)

    # drop irrelevant features
    cleaned_df.drop(['Attrition_Flag', 'CLIENTNUM'], axis=1, inplace=True)

    return cleaned_df


def save_plot(plot, filename):
    '''
    Save a plot to an image file.

    Parameters:
        plot: The plot object to save.
        filename: The filename to save the plot to.
    '''
    plot.savefig(filename, format='jpeg', dpi=300)


def plot_histogram(dataframe, column):
    '''
    Plot histogram for a column and return the plot object.

    Parameters:
        dataframe (DataFrame): Pandas DataFrame containing the data.
        column (str): Name of the column to plot.

    Returns:
        plt.figure: The plot object.
    '''
    # plot the histogram
    plt.figure(figsize=(20, 10))
    dataframe[column].hist(label=column)
    plt.title(f'Histogram of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.legend()
    return plt


def plot_barplot(dataframe, column):
    '''
    Plot barplot for a column and return the plot object.

    Parameters:
        dataframe (DataFrame): Pandas DataFrame containing the data.
        column (str): Name of the column to plot.

    Returns:
        plt.figure: The plot object.
    '''
    plt.figure(figsize=(20, 10))
    dataframe[column].value_counts('normalize').plot(kind='bar')
    plt.title(f'Barplot of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.legend()
    return plt


def plot_density(dataframe, column):
    '''
    Plot density for a column and return the plot object.

    Parameters:
        dataframe (DataFrame): Pandas DataFrame containing the data.
        column (str): Name of the column to plot.

    Returns:
        plt.figure: The plot object.
    '''
    # plot the density plot
    plt.figure(figsize=(20, 10))
    sns.histplot(dataframe[column], stat='density', kde=True)
    plt.title(f'Density plot of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    return plt


def plot_correlation(dataframe):
    '''
    Plots a correlation heatmap for numerical columns of a dataframe and saves as JPEG.

    Parameters:
        dataframe (DataFrame): Pandas DataFrame containing the data.
        folder_name (str): Directory to save the plot.

        Returns:
        plt.figure: The plot object.
    '''
    # plot the correlation heatmap
    numeric_dataframe = dataframe.select_dtypes(include=['number'])
    plt.figure(figsize=(20, 10))
    sns.heatmap(
        numeric_dataframe.corr(),
        annot=False,
        cmap='Dark2_r',
        linewidths=2)
    plt.title('Correlation Heatmap')
    plt.xlabel('Numeric Columns')
    plt.ylabel('Numeric Columns')
    return plt


def perform_eda(dataframe):
    '''
    Perform Exploratory Data Analysis.

    Parameters:
        dataframe (DataFrame): Pandas DataFrame containing the data.
    '''

    # Create eda folder
    folder_name = 'images/eda'
    # Check if the folder already exists
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    else:
        pass

    # Analyze categorical features
    num_columns = dataframe.select_dtypes(exclude='object').columns.tolist()
    for col in num_columns:
        plot = plot_histogram(dataframe, col)
        save_plot(plot, os.path.join(folder_name, f'Histogram_of_{col}.jpeg'))

    # Analyze categorical features and plot distribution
    cat_columns = dataframe.select_dtypes(include='object').columns.tolist()
    for col in cat_columns:
        plot = plot_barplot(dataframe, col)
        save_plot(plot, os.path.join(folder_name, f'Barplot_of_{col}.jpeg'))

    # Analyze density plot of 'Total_Trans_Ct'
    plot_den = plot_density(dataframe, 'Total_Trans_Ct')
    save_plot(
        plot_den,
        os.path.join(
            folder_name,
            f'Density_of_Total_Trans_Ct.jpeg'))

    # Plot correlation heatmap
    plot_heatmap = plot_correlation(dataframe)
    save_plot(
        plot_heatmap,
        os.path.join(
            folder_name,
            f'Correlation Heatmap.jpeg'))


def encoder_helper(dataframe, category_lst, response):
    '''
   Perform target encoding for categorical features.

    Parameters:
        dataframe (DataFrame): Pandas DataFrame containing the data.
        category_lst (list): List of columns that contain categorical features.
        response (str): Name of the response variable.

    Returns:
        DataFrame: DataFrame with target-encoded categorical features.
    '''

    for category in category_lst:
        category_means = dataframe.groupby(category)[response].mean()
        dataframe[category + '_' +
                  response] = dataframe[category].map(category_means)

    return dataframe.drop(category_lst, axis=1)


def perform_feature_engineering(dataframe, response='Churn'):
    '''
    Perform feature engineering.

    Parameters:
        dataframe (DataFrame): Pandas DataFrame containing the data.
        response (str): Name of the response variable.

    Returns:
        tuple: x_train, x_test, y_train, y_test
    '''
    # list of categorical features to be encoded
    cat_columns = dataframe.select_dtypes(include='object').columns.tolist()

    # turn each categorical column into a new column with propotion of churn
    # for each category
    dataframe = encoder_helper(dataframe, cat_columns, response)

    # Seggragate the independent and target variables
    y = dataframe[response]
    x = dataframe.drop(response, axis=1)

    # train test split
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=42)

    return x_train, x_test, y_train, y_test


def classification_report_helper(model_name,
                                 y_train,
                                 y_test,
                                 y_train_preds,
                                 y_test_preds):
    '''
    develops classification report for training and testing results and stores
    report as image in images folder. this helper functin will later be used for classification_report_image fucntion.

    Parameters:
        model_name: (str) name of the model, ie 'Random Forest'
        y_train: training response values
        y_test:  test response values
        y_train_preds: training predictions from model_name
        y_test_preds: test predictions from model_name
    '''
    # Create image folder
    folder_name = 'images/results'
    # Check if the folder already exists
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    else:
        pass

    plt.rc('figure', figsize=(5, 5))

    # Plot Classification report on Train dataset
    plt.text(0.01, 1.25,
             str(f'{model_name} Train'),
             {'fontsize': 10},
             fontproperties='monospace'
             )
    plt.text(0.01, 0.05,
             str(classification_report(y_train, y_train_preds)),
             {'fontsize': 10},
             fontproperties='monospace'
             )

    # Plot Classification report on Test dataset
    plt.text(0.01, 0.6,
             str(f'{model_name} Test'),
             {'fontsize': 10},
             fontproperties='monospace'
             )
    plt.text(0.01, 0.7,
             str(classification_report(y_test, y_test_preds)),
             {'fontsize': 10},
             fontproperties='monospace'
             )

    plt.axis('off')

    # Save figure to ./images folder
    fig_name = f'Classification_report_{model_name}.png'
    save_plot(plt, os.path.join(folder_name, fig_name))


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores
    report as image in images folder using classification_report_helper function

    Parameters:
        y_train: training response values
        y_test:  test response values
        y_train_preds_lr: training predictions from logistic regression
        y_train_preds_rf: training predictions from random forest
        y_test_preds_lr: test predictions from logistic regression
        y_test_preds_rf: test predictions from random forest

    output:
                     None
    '''
    # plot the classification report for Logistic Regression
    classification_report_helper('Logistic Regression',
                                 y_train,
                                 y_test,
                                 y_train_preds_lr,
                                 y_test_preds_lr)

    plt.close()

    # plot the classification report for Random Forest
    classification_report_helper('Random Forest',
                                 y_train,
                                 y_test,
                                 y_train_preds_rf,
                                 y_test_preds_rf)
    plt.close()


def plot_roc_curve(y_test, y_pred_prob):
    '''
    creates Receiver Operating Characteristic (ROC) Curve
    Parameters:
        y_test:  test response values
        y_pred_prob: Probability of positive class
    '''
    # Create image folder
    folder_name = 'images/results'
    # Check if the folder already exists
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    else:
        pass

    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)  # false positive rate, true postive rate

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    # plt.show()
    # Save figure to images folder
    fig_name = f'ROC_curve.png'
    save_plot(plt, os.path.join(folder_name, fig_name))


def train_models(x_train, x_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    Parameters:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    # Create model folder
    folder_name = 'models'
    # Check if the folder already exists
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    else:
        pass

    # Initialize Random Forest model
    rfc = RandomForestClassifier(random_state=42)

    # Initialize Logistic Regression model
    lrc = LogisticRegression(solver='saga', max_iter=3000)
    # Use a different solver if the default 'lbfgs' fails to converge
    # Reference:
    # https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression

    # grid search for random forest parameters and instantiation
    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['log2', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)

    # Train Ramdom Forest using GridSearch
    cv_rfc.fit(x_train, y_train)

    # Train Logistic Regression
    lrc.fit(x_train, y_train)

    # get predictions
    y_train_preds_rf = cv_rfc.best_estimator_.predict(x_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(x_test)

    y_train_preds_lr = lrc.predict(x_train)
    y_test_preds_lr = lrc.predict(x_test)

    # calculate classification scores
    classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf)

    # Plot ROC curve
    y_pred_prob_lrc = lrc.predict_proba(x_test)[:, 1]
    y_pred_prob_rfc = cv_rfc.best_estimator_.predict_proba(x_test)[:, 1]

    plot_roc_curve(y_test, y_pred_prob_lrc)
    plot_roc_curve(y_test, y_pred_prob_rfc)

    # saving the best model
    model_pth = ['models/rfc_model.pkl', 'models/logistic_model.pkl']
    for model, model_pth in zip([cv_rfc.best_estimator_, lrc],
                                model_pth
                                ):
        joblib.dump(model, model_pth)


def feature_importance_plot(model, X_data, model_name):
    '''
    creates and stores the feature importances in pth
    Parameters:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
    '''
    # Create image folder
    folder_name = 'images/results'
    # Check if the folder already exists
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    else:
        pass

    # Calculate feature importances
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title(f"Feature Importance for {model_name}")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)

    # Save figure to output_pth
    fig_name = f'feature_importance_{model_name}.png'
    save_plot(plt, os.path.join(folder_name, fig_name))


def shap_explainer(model, model_name, x_test):
    '''
    creates and stores shap explainer
    Parameters:
            model: model object containing feature_importances_
            model_name: model name (str)
            x_test: test dataset of x
    '''
    # Create image folder
    folder_name = 'images/results'
    # Check if the folder already exists
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    else:
        pass

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_test)
    shap.summary_plot(shap_values, x_test, plot_type="bar")
    # Save figure to images folder
    fig_name = f'Shap_Explainer_{model_name}.png'
    save_plot(plt, os.path.join(folder_name, fig_name))


if __name__ == "__main__":
    # Load dataset
    dataset = import_data("data/bank_data.csv")
    print('Dataset successfully loaded')
    if dataset is None:
        print('Dataset is not found')
        exit()

    # Clean data
    cleaned_data = data_cleaning(dataset)
    print('Finished Data Cleaning')

    # importing matplotlib and setting the backend to 'Agg'.
    '''
    The Qt used as the backend for rendering the plots in a GUI environment is required when creating plots using matplotlib.
    However, since the terminal environment where this code was run does not have any GUI capability,
    this will encounter warnings related to Qt when trying to create plots.
    This will delay the running time significantly.

    The 'Agg' backend is a non-interactive backend suitable for generating image files without requiring a GUI.
    '''
    import matplotlib
    matplotlib.use('Agg')

    # Perform Exploratory Data Analysis
    print('Started EDA')
    perform_eda(cleaned_data)
    print("Finished EDA. All plots are saved at the 'images/eda' folder")

    # Perform feature engineering
    train_x, test_x, train_y, test_y = perform_feature_engineering(
        cleaned_data, response='Churn')
    print('Finished Feature Engineering')

    # Train models
    print('Started training models. This may take up to 15-20 minutes to run')
    train_model = train_models(train_x, test_x, train_y, test_y)
    print("Training completed. Best model and performance metrics saved. Metrics images are saved at the 'images/results' folder")

    # Load models
    rfc_model = joblib.load('./models/rfc_model.pkl')
    lr_model = joblib.load('./models/logistic_model.pkl')
    print('Finished models loading')

    # Plot feature importance
    feature_importance_plot(
        rfc_model,
        train_x,
        'Random_Forest')
    print("Feature Importance Plot has been created and saved at the 'images/results' folder")

    # Plot Shap Explainer
    shap_explainer(rfc_model, 'Random Forest', test_x)
    print("Shap explainer plot has been created and saved at the 'images/results' folder")
