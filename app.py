# this line brings streamlit library, and lets you call it using 
# the short name st
import streamlit as st

# this has to be the first streamlit command in the file because it 
# tells the app to use the browser's full width instead of a narrow 
#centered column
st.set_page_config(layout="wide")

# display a big title at the top of the app page
st.title("Medical Anomaly & Model Steering Studio")

import pandas as pd

df = pd.read_csv("https://raw.githubusercontent.com/Ruohan-Yang/Heart-Disease-Data-Set/refs/heads/main/processed.cleveland.csv")

df = df[(df['ca'] != '?') & (df['thal'] != '?')]
df['ca'] = df['ca'].astype(int)
df['thal'] = df['thal'].astype(int)

st.header("Dataset")
with st.expander("View raw data"):
    st.write(df.head()) #st.write(...) is Streamlit's just show me this command 

# Step 2: Scale + PCA + IsolationForest

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

X = df.drop('target', axis=1)
y = df['target']

X_scaled = StandardScaler().fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

iso_forest = IsolationForest(contamination=0.1, random_state=42)
predictions = iso_forest.fit_predict(X_scaled)

st.header("Dimensionality Reduction")
st.write("PCA variance explained: ", pca.explained_variance_ratio_)

# scatter plot
import plotly.express as px

#plot_df is to get a small new table just for plotting the 2 PCA coordinates plus a readble anomlay labe
plot_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
plot_df['anomaly'] = predictions

#.map({1: 'Normal', -1: 'Anomaly'}) converts the 1/-1 into words, since a legend showing "Normal"/"Anomaly" is more readable
plot_df['anomaly'] = plot_df['anomaly'].map({1: 'Normal', -1: 'Anomaly'})

if "reset_counter" not in st.session_state: 
    st.session_state.reset_counter = 0

if "y" not in st.session_state:
    st.session_state.y = y.copy()

from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(max_depth=3, random_state= 42)
tree.fit(X_scaled, st.session_state.y)

#from sklearn.tree import export_text

#st.subheader("Current Decision Rules")
#st.text(export_text(tree, feature_names=list(X.columns)))

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

st.subheader("Current Decision Rules")

fig_tree, ax = plt.subplots(figsize=(20, 10))
plot_tree(tree, feature_names=list(X.columns), class_names=True, filled=True, rounded=True, ax=ax)
st.pyplot(fig_tree)


#px.scatter(...) is what builds the interactive chart
fig = px.scatter(plot_df, x='PC1', y= 'PC2', color= 'anomaly', title= 'Patient Clusters (PCA + IsolationForest)')

st.header("Steering Panel")
left_col, right_col = st.columns([1,2])

with right_col:
    event = st.plotly_chart(fig, on_select="rerun", selection_mode=("points", "box", "lasso"), key=f"scatter_{st.session_state.reset_counter}")

with left_col:
    if "show_success" in st.session_state:
        st.success(st.session_state.show_success)
        del st.session_state.show_success
    
    with st.expander("Debug: raw selection data"):
        st.write(event.selection)

    selected_indices = event.selection.point_indices

    if selected_indices:
        st.write(f"You selected {len(selected_indices)} patient(s).")
        st.write(plot_df.iloc[selected_indices])

        if st.button("Clear Selection"):
            st.session_state.reset_counter += 1
            st.rerun()

        new_label = st.radio("Relabel selected patients as:", ["Normal", "Anomaly"])

        if st.button("Apply Feedback"):
            new_value = 0 if new_label == "Normal" else 1
            st.session_state.y.iloc[selected_indices] = new_value

            tree.fit(X_scaled, st.session_state.y)

            st.session_state.reset_counter += 1
            st.session_state.show_success = f"Updated {len(selected_indices)} patient(s) and retrained the model."
            st.rerun()


