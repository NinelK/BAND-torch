import pandas as pd

# Read the CSV file
df = pd.read_csv('dataset_summary.csv')

#split 'Dataset' into Monkey and Date
df[['Monkey','Date']] = df['Dataset'].str.split('_CO_FF_',expand=True)
#remove ".mat" from Date
df['Date'] = df['Date'].str.replace('.mat','')
# remove 'Dataset' column
df.drop(columns=['Dataset'],inplace=True)
# place 'Monkey' and 'Data' in first columns
cols = df.columns.tolist()
cols = cols[-2:] + cols[:-2]
df = df[cols]

# Get column names from the DataFrame
columns = df.columns

# Define the LaTeX table header
latex_table_header = r'''\begin{table}[htbp]
\centering
\resizebox{\textwidth}{!}{%
\begin{tabular}{lc|''' + 'cccc|' * (len(columns) - 3) + r'''}
\hline
'''

# Generate the column formatting for the LaTeX table
column_formatting = ''.join([f"{col} & " for col in columns[:-1]])
latex_table_header += column_formatting + columns[-1] + r' \\ \hline' + '\n'

# Initialize the LaTeX table content
latex_table_content = ''

# Append each row of the DataFrame to the LaTeX table
for index, row in df.iterrows():
    row_content = ' & '.join([str(val) for val in row]) + r' \\ \hline' + '\n'
    latex_table_content += row_content

# Define the LaTeX table footer
latex_table_footer = r'''\end{tabular}%
}
\caption{ Variance explained by a XXX decoder given neural recordings from M1, PMd, or both areas together ($R^2$, \%)}
\label{tab:XXX_decoder_pred}
\end{table}'''

# Combine header, content, and footer to form the complete LaTeX table
latex_table = latex_table_header + latex_table_content + latex_table_footer

# Write the LaTeX table to a file
with open('output_table.tex', 'w') as f:
    f.write(latex_table)