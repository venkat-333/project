import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import math
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

# Title
st.title("Premier League Analysis")
st.image("https://cdnuploads.aa.com.tr/uploads/Contents/2023/01/13/thumbs_b_c_7982afecfc0cefec23851b81850809c9.jpg?v=165134",width=800)

# Sidebar
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Load dataset
    mm = pd.read_csv(uploaded_file)
    st.subheader("Dataset Overview")
    st.write(mm)



    if st.checkbox("Show Dataset Description"):
        st.write(mm.describe())

    if st.checkbox("Show Missing Values"):
        st.write(mm.isnull().sum())

    # Data Cleaning
    mm['xG'] = mm['xG'].fillna(mm['xG'].mean())
    mm['Gls'] = mm['Gls'].fillna(mm['Gls'].mean())
    mm = mm.dropna()

    # Sidebar selections for visualizations
    st.sidebar.header("Data Visualization")
    if st.sidebar.checkbox("Heatmap of Missing Values"):
        st.subheader("Heatmap of Missing Values")
        sns.heatmap(mm.isnull(), cbar=False)
        st.pyplot()

 # Group players by position and plot
    if st.sidebar.checkbox('Player Positions'):
        st.subheader("Barplot of players position")
        var1 = mm.groupby('Pos').size().sort_values(ascending=False)
        fig1, ax1 = plt.subplots(figsize=(13, 6))
        var1.head(630).plot(kind='bar', ax=ax1, color=sns.color_palette('husl'))
        ax1.set_title('Position of Players')
        st.pyplot(fig1)

    # Calculate and display statistics
    tg = math.trunc(mm['Gls'].sum())
    ta = math.trunc(mm['Ast'].sum())
    ps = math.trunc(mm['PK'].sum())
    pa = math.trunc(mm['PKatt'].sum())

    st.header('Key Statistics')
    st.write(f"**Total Goals Scored:** {tg}")
    st.write(f"**Total Assists Scored:** {ta}")
    st.write(f"**Total Penalties Scored:** {ps}")
    st.write(f"**Total Penalties Attempted:** {pa}")

    if st.sidebar.checkbox('EPL Player Stats Visualization'):
       st.subheader("Scatter plot of Goals vs Assists")
       fig2, vv= plt.subplots()
       vv= plt.scatter(mm['Gls.1'],mm['Ast.1'])
       st.pyplot(fig2)

    # Age distribution plot
    if st.sidebar.checkbox('Age Distribution of Players'):
      st.subheader("Barplot of age distribution of the players")
      age = mm.groupby('Age').size().sort_values(ascending=True)
      fig3, ax2 = plt.subplots(figsize=(13, 6))
      age.head(630).plot(kind='bar', ax=ax2, color=sns.color_palette('magma'))
      ax2.set_title('Age of Players')
      st.pyplot(fig3)

    # Under-20 players and their teams
    if st.sidebar.checkbox('Teams with Under-20 Players'):
        st.subheader("Barplot of players under 20")
        under20 = mm[mm['Age'] < 20]
        fig4, ax3 = plt.subplots(figsize=(12, 6))
        under20['Team'].value_counts().plot(kind='bar', ax=ax3, color=sns.color_palette('cubehelix'), edgecolor='black')
        ax3.set_title('Teams with Players Under 20')
        st.pyplot(fig4)
    # Goals and assists pie chart
    if st.sidebar.checkbox('Goals and Assists Distribution'):
      st.subheader("Pie chart of goals & assists goals distribution")
      assists = mm['Ast'].sum()
      goals = mm['Gls'].sum()
      data = [goals - assists, assists]
      labels = ['Goals without assists', 'Goals with assists']
      colors = sns.color_palette('Set2')
      fig5, ax4 = plt.subplots()
      ax4.pie(data, labels=labels, colors=colors, autopct='%.0f%%')
      st.pyplot(fig5)

    if st.sidebar.checkbox('Pairplot varition '):
       st.subheader("Pairplot of EPL")
       fig6 = sns.pairplot(mm)
       st.pyplot(fig6)


    # Display unique teams
    if st.sidebar.checkbox('List of Unique Teams'):
       unique_teams = mm['Team'].unique()
       st.write(unique_teams)

    # Team with most goals
    if st.sidebar.checkbox('Team with Most Goals'):
      team_goals = mm.groupby('Team')['Gls'].sum()
      team_with_most_goals = team_goals.idxmax()
      most_goals = team_goals.max()
      st.write(f"The team with the most goals is: **{team_with_most_goals}** with **{most_goals}** goals.")

    # Player with most goals
    if st.sidebar.checkbox('Player with Most Goals'):
      top_scorer = mm.loc[mm['Gls'].idxmax()]
      st.write(f"The player who scored the most goals is: **{top_scorer['Player']}**")
      st.write(f"Goals scored: **{top_scorer['Gls']}**")


    # Feature Extraction
    st.sidebar.header("Feature Extraction and Model Training")
    x = mm.drop(columns='Gls')
    y = mm['Gls']

    # Data split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)

    # Initialize accuracy variables
    lr_accuracy = None
    dt_accuracy = None
    nb_accuracy = None
    pr_accuracy = None

    if st.sidebar.checkbox("Run Logistic Regression"):
        st.subheader("Logistic Regression")
        categorical_cols = x_train.select_dtypes(include=['object']).columns
        numerical_cols = x_train.select_dtypes(include=['number']).columns

        numeric_transformer = SimpleImputer(strategy='mean')
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numerical_cols),
                ('cat', categorical_transformer, categorical_cols)
            ]
        )

        model_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(max_iter=1000))
        ])

        y_train_binned = pd.cut(y_train, bins=3, labels=["low", "medium", "high"])
        model_pipeline.fit(x_train, y_train_binned)
        y_pred_lr = model_pipeline.predict(x_test)
        lr_accuracy = accuracy_score(pd.cut(y_test, bins=3, labels=["low", "medium", "high"]), y_pred_lr)
        st.write(f"Logistic Regression Accuracy: {lr_accuracy}")

    if st.sidebar.checkbox("Run Decision Tree"):
        st.subheader("Decision Tree")
        label_encoder = LabelEncoder()
        x_train = x_train.apply(lambda col: label_encoder.fit_transform(col) if col.dtypes == 'object' else col)
        x_test = x_test.apply(lambda col: label_encoder.fit_transform(col) if col.dtypes == 'object' else col)

        dtc = DecisionTreeClassifier()
        dtc.fit(x_train, y_train)
        y_pred_dt = dtc.predict(x_test)
        dt_accuracy = accuracy_score(y_test, y_pred_dt)
        st.write(f"Decision Tree Accuracy: {dt_accuracy}")

    if st.sidebar.checkbox("Run Naive Bayes"):
        st.subheader("Naive Bayes")
        imputer = SimpleImputer(strategy='mean')
        x_train_imputed = imputer.fit_transform(x_train)
        x_test_imputed = imputer.transform(x_test)

        nb = GaussianNB()
        nb.fit(x_train_imputed, y_train)
        y_pred_nb = nb.predict(x_test_imputed)
        nb_accuracy = accuracy_score(y_test, y_pred_nb)
        st.write(f"Naive Bayes Accuracy: {nb_accuracy}")

    if st.sidebar.checkbox("Run PCA"):
        st.subheader("PCA")
    
    mm['xG'] = mm['xG'].fillna(mm['xG'].mean())
    mm['Gls'] = mm['Gls'].fillna(mm['Gls'].mean())

    x = mm.drop(columns='Gls')
    y = mm['Gls']

    # Train-test split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)

    # Identify numerical and categorical columns
    numerical_cols = x_train.select_dtypes(include=np.number).columns
    categorical_cols = x_train.select_dtypes(include='object').columns

    # Preprocessing pipelines
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Combine transformations
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )

    # Transform data
    x_train_preprocessed = preprocessor.fit_transform(x_train)
    x_test_preprocessed = preprocessor.transform(x_test)

    # Apply PCA
    pca = PCA(n_components=0.95)  # Retain 95% of the variance
    x_train_pca = pca.fit_transform(x_train_preprocessed)
    x_test_pca = pca.transform(x_test_preprocessed)

    # Fit KMeans on PCA-transformed data
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(x_train_pca)

    # Predict clusters
    train_clusters = kmeans.predict(x_train_pca)
    test_clusters = kmeans.predict(x_test_pca)

    # Train a Logistic Regression model
    model = LogisticRegression()
    model.fit(x_train_pca, y_train)

    # Make predictions
    y_pred_lr = model.predict(x_test_pca)
    y_pred_prob_lr = model.predict_proba(x_test_pca)[:, 1]

    # Convert probability predictions to class labels
    if y_test.dtype in ['float64', 'float32']:  
        y_test = (y_test > 0.5).astype(int)  # Convert to binary classes (if binary classification)

    # Compute accuracy
    pr_accuracy = accuracy_score(y_test, y_pred_lr)

    # Compute AUC only for classification problems
    if len(set(y_test)) > 2:  # Multi-class classification
        lr_auc = roc_auc_score(y_test, model.predict_proba(x_test_pca), multi_class='ovr')
    else:  # Binary classification
        lr_auc = roc_auc_score(y_test, y_pred_prob_lr)


    st.write(f"Model Accuracy: {pr_accuracy:.4f}")
   
    # Comparison
    st.subheader("Model Comparison")
    model_comparison = pd.DataFrame({
        'Model': ['Logistic Regression', 'Decision Tree', 'Naive Bayes','PCA'],
        'Accuracy': [lr_accuracy, dt_accuracy, nb_accuracy,pr_accuracy]
    })
    model_comparison = model_comparison.dropna()
    st.write(model_comparison)

    plt.figure(figsize=(8, 5))
    sns.barplot(x='Model', y='Accuracy', data=model_comparison, palette='viridis')
    plt.title('Model Accuracy Comparison')
    st.pyplot()