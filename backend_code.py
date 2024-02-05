inp_date = 2019
import sys
import cmath
import numpy as np
import pandas as pd
import csv
import time
import subprocess
from xml.etree.ElementTree import Element, ElementTree, SubElement
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull
import geopandas as gpd
from shapely.geometry import Polygon
import re
import os


# Start time calculation
start1 = time.time()


# Breaking a 1 point of longitude and latitude by n distance ahead and behind
# Here n is 5 meter as org pixel is 10 by 10meter
def new_coordinates(df, n, i):
    dx = n
    dy = n
    r_earth = 6371000
    lat_data = df['latitude'].tolist()

    long_data = df['longitude'].tolist()
    pixel_columns = f'pix_{i}'
    pix = df[pixel_columns].tolist()

    new_coords = []

    for i in range(len(lat_data)):
        new_latitude_plus_5 = np.real(lat_data[i] + (dy / r_earth) * (180 / cmath.pi))
        new_latitude_minus_5 = np.real(lat_data[i] - (dy / r_earth) * (180 / cmath.pi))
        new_longitude_plus_5 = np.real(long_data[i] + (dx / r_earth) * (180 / cmath.pi))
        new_longitude_minus_5 = np.real(long_data[i] - (dx / r_earth) * (180 / cmath.pi))

        new_coords.append({'latitude': new_latitude_plus_5, 'longitude': new_longitude_plus_5, pixel_columns: pix[i]})
        new_coords.append({'latitude': new_latitude_plus_5, 'longitude': new_longitude_minus_5, pixel_columns: pix[i]})
        new_coords.append({'latitude': new_latitude_minus_5, 'longitude': new_longitude_plus_5, pixel_columns: pix[i]})
        new_coords.append({'latitude': new_latitude_minus_5, 'longitude': new_longitude_minus_5, pixel_columns: pix[i]})

    new_df = pd.DataFrame(new_coords)
    new_excel_file = f'{pixel_columns}_coords.csv'
    new_df.to_csv(new_excel_file, index=False)


# Pivoting the orignal csv in 2D for interpolation
def pivoting_orignal_files(in_file, i):
    # in_file = pd.read_csv('pix_1.csv')
    # Sort the DataFrame by 'latitude' in descending order (highest to lowest)
    # in_file = in_file.sort_values(by='longitude', ascending=True)
    # in_file = in_file.sort_values(by='latitude', ascending=True)
    pixel_value = f'pix_{i}'

    # Pivot the table with 'latitude' as the index, 'longitude' as columns, and aggregate with a function like 'first' to keep unique values
    pivoted_df = pd.pivot_table(in_file, index='latitude', columns='longitude', values=pixel_value)
    pivoted_df = pivoted_df.sort_values(by='latitude', ascending=False)

    # Extract the values only and filling blanks with 0
    pivoted_df = pivoted_df.iloc[:, 1:]
    pivoted_df.fillna(-1, inplace=True)

    # Save the extracted data to a new CSV file
    pivoted_df.to_csv(f'destination_{i}.csv', index=False, header=False)


# Interpolates the file having 2D input values of orignal pixel into new values of (x_new*y_new) size
def BilinInterpolated_PixValue(csv_file, x_new, y_new, i):
    try:
        I = []
        with open(csv_file, 'r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                I.append(row)

        [j, k] = np.shape(I)
        x_scale = round(x_new / (j), 3)
        y_scale = round(y_new / (k - 1), 3)
        M = []

        for count1 in range(x_new):
            M_row = []
            for count2 in range(y_new):
                W = -(((count1 / x_scale) - np.floor(count1 / x_scale)) - 1)
                H = -(((count2 / y_scale) - np.floor(count2 / y_scale)) - 1)
                i00_row = int(np.floor(count1 / x_scale))
                i00_col = int(np.floor(count2 / y_scale))
                i01_row = int(np.ceil(count1 / x_scale))
                i01_col = int(np.floor(count2 / y_scale))
                i10_row = int(np.floor(count1 / x_scale))
                i10_col = int(np.ceil(count2 / y_scale))
                i11_row = int(np.ceil(count1 / x_scale))
                i11_col = int(np.ceil(count2 / y_scale))
                # Check if the calculated indices are within the valid range of rows and columns
                if (
                        0 <= i00_row < len(I) and
                        0 <= i00_col < len(I[0]) and
                        0 <= i01_row < len(I) and
                        0 <= i01_col < len(I[0]) and
                        0 <= i10_row < len(I) and
                        0 <= i10_col < len(I[0]) and
                        0 <= i11_row < len(I) and
                        0 <= i11_col < len(I[0])
                ):
                    i00 = float(I[i00_row][i00_col])
                    i01 = float(I[i01_row][i01_col])
                    i10 = float(I[i10_row][i10_col])
                    i11 = float(I[i11_row][i11_col])
                else:
                    i00 = i01 = i10 = i11 = 0.0  # Default to zero
                M_row.append((1 - W) * (1 - H) * i11 + (W) * (1 - H) * i10 + (1 - W) * (H) * i01 + (W) * (H) * i00)
            M.append(M_row)

        df = pd.DataFrame(M)
        excel_file = f'output{i}.csv'
        df.to_csv(excel_file, index=False, header=False)
        # print(f"Data has been exported to '{excel_file}'.")

    except Exception as e:
        print(f"An error occurred: {e}")


# Leaving a row and a column and then pasting the new pix matrix
def leave_row_column(input_file, i):
    try:
        # Read the input CSV file
        with open(input_file, 'r') as csvfile:
            reader = csv.reader(csvfile)
            data = list(reader)

        # Determine the dimensions of the original data
        num_rows = len(data) + 1
        num_columns = len(data[0]) + 1

        # Initialize an empty list to store the shifted data
        shifted_data = [['' for _ in range(num_columns + 1)] for _ in range(num_rows + 1)]

        # Copy the original data to the shifted data, leaving the first row and first column empty
        for row in range(1, num_rows):
            for col in range(1, num_columns):
                shifted_data[row][col] = data[row - 1][col - 1]

        # Write the shifted data to the output CSV file
        with open(input_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(shifted_data)

    except Exception as e:
        print(f"An error occurred: No change detected here")


# Extracting new index and hearder from output of new_coordinates output
def extract_index_header(input_file_path, i):
    # input_file_path = 'pix_1_coords.csv'  # Replace with the path to your input CSV file
    output_long = f'unique_longitudes{i}.csv'
    output_lat = f'unique_latitudes{i}.csv'

    # Read the CSV file into a DataFrame
    df = pd.read_csv(input_file_path)
    # Assuming the column containing coordinates is named 'coordinates_column'
    lat_column = 'latitude'  # Replace with the actual column name
    long_column = 'longitude'

    # Extract unique coordinates from the specified column
    unique_lat = df[lat_column].unique()
    unique_long = df[long_column].unique()

    # Create a new DataFrame with the unique coordinates
    unique_lat_df = pd.DataFrame({'latitude': unique_lat})
    unique_long_df = pd.DataFrame({'longitude': unique_long})
    unique_long_df = unique_long_df.sort_values(by='longitude', ascending=True)

    # Save the unique coordinates to a new CSV file
    unique_lat_df.to_csv(output_lat, index=False)
    unique_long_df.to_csv(output_long, index=False)


# Concatinating the spaced 2D matrix with index and header
def concatinate_index_header(input_lat, input_long, input_file, i):
    # input_lat = f'tempcoordinates_unique_latitudes{i}.csv'
    # input_long = f'tempcoordinates_unique_longitudes{i}.csv'
    # input_file = f'output{i}.csv'

    # Read the input CSV file
    with open(input_lat, 'r') as csvfile:
        reader = csv.reader(csvfile)
        data_lat = list(reader)

    with open(input_long, 'r') as csvfile:
        reader = csv.reader(csvfile)
        data_long = list(reader)

    with open(input_file, 'r') as csvfile:
        reader = csv.reader(csvfile)
        data_input = list(reader)

    for j in range(len(data_input[0]) - 1):
        data_input[0][j] = data_long[j][0]
    for k in range(len(data_input) - 1):
        data_input[k][0] = data_lat[k][0]

    with open(input_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data_input)


# Reverse pivoting
def reverse_pivoting(input_file, a):
    # Read the input CSV file without specifying column names
    # input_file = 'output1.csv'
    df = pd.read_csv(input_file, header=None)

    # Extract the latitude values from the first column
    latitude_values = df.iloc[:, 0].values

    # Extract the longitude values from the first row
    longitude_values = df.iloc[0, 1:].values

    # Extract the data values (pixel values) from the remaining cells
    data_values = df.iloc[1:, 1:].values

    # Create a new DataFrame with the desired columns (lat, long, pix)
    output_df = pd.DataFrame(columns=["lat", "long", "pix"])
    temp = []
    # Populate the new DataFrame with data
    for i in range(len(latitude_values) - 1):
        for j in range(len(longitude_values) - 1):
            temp.append({"lat": latitude_values[i + 1], "long": longitude_values[j], "pix": data_values[i][j]})
    new_df = pd.DataFrame(temp)
    output_file = f'output_final_{a}.csv'
    new_df.to_csv(output_file, index=False)

    # print(f'Converted data saved to {output_file}')

# Not in use
def clustering(area, original_df, i):
    # Filter 1
    filtered_df1 = original_df[(original_df['pix'] > -0.1) & (original_df['pix'] != -1)]

    # Filter 2
    filtered_df2 = original_df[(original_df['pix'] <= -0.1) & (original_df['pix'] > -0.2) & (original_df['pix'] != -1)]

    # Filter 3
    filtered_df3 = original_df[(original_df['pix'] <= -0.2) & (original_df['pix'] > -0.3) & (original_df['pix'] != -1)]

    # Filter 4
    filtered_df4 = original_df[(original_df['pix'] <= -0.3) & (original_df['pix'] > -0.4) & (original_df['pix'] != -1)]

    # Filter 5
    filtered_df5 = original_df[(original_df['pix'] <= -0.4) & (original_df['pix'] != -1)]

    # Save the filtered dataframes to separate CSV files
    # filtered_df1.to_csv(f'{i}{area}_filter_1.csv', index=False)
    # filtered_df2.to_csv(f'{i}{area}_filter_2.csv', index=False)
    # filtered_df3.to_csv(f'{i}{area}_filter_3.csv', index=False)
    # filtered_df4.to_csv(f'{i}{area}_filter_4.csv', index=False)
    # filtered_df5.to_csv(f'{i}{area}_filter_5.csv', index=False)


# Combining all output
def combine_output_final(n,file_name):
    csv_files = []
    for i in range(n - 2):
        csv_files.append(f'output_final_{i + 1}.csv')
    df1 = pd.read_csv('output_final_1.csv')
    lat = df1['lat']
    long = df1['long']
    combined_df = pd.DataFrame({'latitude': lat, 'longitude': long})
    i = 0
    for csv_file in csv_files:
        # Read the CSV file into a separate DataFrame
        df = pd.read_csv(csv_file)
        column_array = df['pix'].to_numpy()
        combined_df[f'pix_{i + 1}'] = column_array
        i = i + 1
    csv_file_path = f'aa_{file_name}.csv'
    combined_df.to_csv(csv_file_path, index=False)
    # Filling NaN values with 0
    df = pd.read_csv(csv_file_path)
    df.fillna(0, inplace=True)
    df.to_csv(csv_file_path, index=False)


# Change header
def preparing_aa(source_csv, target_csv):
    header_df = pd.read_csv(source_csv, nrows=0)
    target_df = pd.read_csv(target_csv)
    target_df.columns = header_df.columns
    target_df.to_csv(target_csv, index=False)


# Detecting the date where it crossed threshold value
def change_detection(input_file, output_file):
    try:
        with open(input_file, 'r') as csvfile:
            reader = csv.reader(csvfile)
            data_input = list(reader)

        lat = []
        long = []
        pix = []
        date = []
        # print(data_input)
        for i in range(1, len(data_input) - 1):  # 1 to 161
            for j in range(2, len(data_input[0]) - 1):  # 2 to 41
                if float(data_input[i][j]) > -0.1 and float(data_input[i][j]) != -1 and float(
                        data_input[i][j]) != 0.0 and float(data_input[i][len(data_input[0]) - 1]) > -0.2:
                    if j <= (len(data_input[0]) - 1) - 3:
                        ctr = 1
                        sum_average = float(data_input[i][j])
                        for k in range(j + 1, len(data_input[0]) - 1):
                            # if float(data_input[i][k]) > 0:
                            #     break
                            sum_average = sum_average + float(data_input[i][k])
                            ctr = ctr + 1
                        average = sum_average / ctr
                        if average >= -0.15:
                            lat.append(data_input[i][0])
                            long.append(data_input[i][1])
                            pix.append(data_input[i][j])
                            date.append(data_input[0][j])

                        break

        if not lat:  # If the list is empty, no change is detected
            print("No change detected.")
        else:
            data = {'Latitude': lat, 'Longitude': long, 'Pixel': pix, 'Date': date}
            df = pd.DataFrame(data)
            df.to_csv(output_file, index=False)
            # print(f'Data saved to {output_file}')

    except Exception as e:
        print(f"An error occurred: {e}")


# Total area calculation in acres
def area_calc(input_file, side_lenght):
    df = pd.read_csv(input_file)
    num_rows = len(df)
    total_area = num_rows * 0.000247 * (side_lenght) ** 2
    print(f'Area In Acres = {total_area}')
    return total_area

# Converting a csv file to kml file
def csv_to_kml(csv_file, kml_file):
    kml = Element('kml')
    document = SubElement(kml, 'Document')

    with open(csv_file, 'r') as csvfile:
        csv_reader = csv.DictReader(csvfile)
        for row in csv_reader:
            placemark = SubElement(document, 'Placemark')
            # name = SubElement(placemark, 'name')
            # name.text = row['Pixel']  # Replace with your CSV column name
            description = SubElement(placemark, 'description')
            description.text = row['Date']  # Replace with your CSV column name
            point = SubElement(placemark, 'Point')
            coordinates = SubElement(point, 'coordinates')
            coordinates.text = f"{row['Longitude']},{row['Latitude']},0"  # Replace with your CSV column names for longitude and latitude

    tree = ElementTree(kml)
    tree.write(kml_file, encoding='utf-8', xml_declaration=True)


# Filtering according to the dates
# Update the date over here
def date_filter(inp,area_name,file_name, inp_date):
    # Read the CSV file
    df = pd.read_csv(inp)
    # Convert the 'date' column to datetime objects
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')  # 'coerce' handles invalid date strings gracefully

    # Filter values after the year 2021
    filtered_df = df[df['Date'].dt.year >= inp_date]

    # Filter values for a specific month (e.g., January)
    # filtered_df = filtered_df[filtered_df['Date'].dt.month >= 10]  # 1 corresponds to January
    # Specify the path to save the filtered CSV file
    filtered_csv_file = f"1_{area_name}_{file_name}_detected_data.csv"

    # Save the filtered data to a new CSV file
    filtered_df.to_csv(filtered_csv_file, index=False)
    # print(f"Build-up Detected data saved to {filtered_csv_file}.")


# Clustering for cleaned_data
def clustering(input_data,file_name):
    # Load your CSV data into a DataFrame
    data = pd.read_csv(input_data)

    # Assuming your CSV contains 'latitude' and 'longitude' columns
    coordinates = data[['Latitude', 'Longitude']]

    # Define the DBSCAN model
    epsilon = 0.0003  # Adjust this value based on the spatial density of your data
    min_samples = 40  # Adjust this value based on your data and the level of noise you want to eliminate
    dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)

    # Fit the DBSCAN model to your data
    data['cluster'] = dbscan.fit_predict(coordinates)

    # Noise points will have a cluster label of -1, and valid points will have a positive integer cluster label

    # Save the data with noise points removed to a new CSV file
    cleaned_data = data[data['cluster'] != -1]
    cleaned_data.to_csv(f'cleaned_data_{file_name}.csv', index=False)


# Preparing cleaned data for 1 cluster - 1 date
def cluster_date(input_clean,file_name):
    data = pd.read_csv(input_clean)

    max_cluster = data['cluster'].max()

    df = pd.DataFrame(data)

    for i in range(max_cluster + 1):
        # Find the minimum date for cluster 0
        min_date_cluster_i = df[df['cluster'] == i]['Date'].min()
        print(i)
        # Assign the minimum date to all rows in cluster 0
        df.loc[df['cluster'] == i, 'Date'] = min_date_cluster_i
    data.to_csv(f'cleaned_data_{file_name}.csv')


# Creating convex hull for each cluster
def Convex_hull(input_clean):
    # Load your clustered data from the provided input filename
    data = pd.read_csv(input_clean)

    # Convert the 'Date' column to datetime format with the correct format
    data['Date'] = pd.to_datetime(data['Date'], format='%d-%m-%Y')

    # Get the unique cluster labels from the 'cluster' column
    unique_clusters = data['cluster'].unique()

    # Iterate through each unique cluster label and find its convex hull
    for cluster_label in unique_clusters:
        cluster_data = data[data['cluster'] == cluster_label]
        cluster_points = cluster_data[['Latitude', 'Longitude', 'Date']].to_numpy()

        # Convert the 'Date' column to a numeric format (e.g., timestamp or ordinal)
        cluster_points[:, 2] = cluster_data['Date'].apply(lambda x: x.timestamp()).to_numpy()

        try:
            # Compute the convex hull for the cluster with the 'QJ' option
            hull = ConvexHull(cluster_points, qhull_options='QJ')

            # Get the vertices of the convex hull for this cluster
            hull_vertices = cluster_points[hull.vertices]

            # Create a DataFrame for this cluster's convex hull
            cluster_hull_data = pd.DataFrame(hull_vertices, columns=['Latitude', 'Longitude', 'Date'])

            # Convert the 'Date' column back to datetime format if needed
            cluster_hull_data['Date'] = pd.to_datetime(cluster_hull_data['Date'], unit='s')

            # Save the convex hull data to a separate CSV file
            filename = f'convex_hull_cluster_{cluster_label}.csv'
            cluster_hull_data.to_csv(filename, index=False)

        except Exception as e:
            print(f"Error for cluster {cluster_label}: {str(e)}")


# The final shape file
def create_convex_hull_shapefile(input, number,fileName):
    # Load your clustered data from a CSV file or any other source
    data = pd.read_csv(input)

    # Get the unique cluster labels from the 'cluster' column
    unique_clusters = data['cluster'].unique()

    # Create an empty GeoDataFrame to store the convex hulls
    gdf = gpd.GeoDataFrame()

    # Iterate through each unique cluster label and find its convex hull
    for cluster_label in unique_clusters:
        cluster_data = data[data['cluster'] == cluster_label]
        cluster_points = cluster_data[['Longitude', 'Latitude']].to_numpy()  # Note the order: (lon, lat)

        # Compute the convex hull for the cluster
        hull = ConvexHull(cluster_points)


        # Get the vertices of the convex hull for this cluster
        hull_vertices = cluster_points[hull.vertices]

        # Create a Shapely Polygon from the convex hull vertices
        polygon = Polygon(hull_vertices)

        attributes = {
            "name": f"{number}",
            "area": polygon.area,
            "perimeter": polygon.length
        }

        # Create a GeoDataFrame for this cluster's convex hull
        cluster_gdf = gpd.GeoDataFrame({'geometry': [polygon]}, crs='EPSG:4326')

        # Add the date as an attribute to the polygon
        date = cluster_data['Date'].iloc[0]
        cluster_gdf['Start Date'] = date

        # Concatenate this cluster's GeoDataFrame to the overall GeoDataFrame
        gdf = pd.concat([gdf, cluster_gdf], ignore_index=True)

    # Set the GeoDataFrame's coordinate reference system (CRS)
    gdf.crs = 'EPSG:4326'  # WGS 84

    # Save the GeoDataFrame as an ESRI Shapefile
    gdf.to_file(f'{fileName}_{number}.shp')


# Deleting unnecessary files
def delete_un():
    current_dir = os.path.dirname(__file__)
    prefixes_to_delete = ["destination", "output", "pix", "unique"]
    files = os.listdir(current_dir)
    for file in files:
        for prefix in prefixes_to_delete:
            if file.startswith(prefix):
                file_path = os.path.join(current_dir, file)
                os.remove(file_path)
                # print(f"Deleted: {file_path}")



# Main function
# Calling the input  file from frontend to backend code
if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python backend.py <csv_file_path>")
    else:
        csv_file_path = sys.argv[1]
        # process_csv(csv_file_path)

#print(year)
area_name = 'Area'
# print(csv_file_path)
# Get the filename without extension
file_name = os.path.splitext(os.path.basename(csv_file_path))[0]
# Print the variable name
# print(file_name)
input_file = csv_file_path
output_file = f'{area_name}_input.csv'
df = pd.read_csv(input_file)
column_mapping = {'latitude': 'latitude', 'longitude': 'longitude'}
# Rename the other columns to 'pix_1', 'pix_2', 'pix_3', etc.
for i, column in enumerate(df.columns[2:]):
    column_mapping[column] = f'pix_{i + 1}'
df.rename(columns=column_mapping, inplace=True)
df.to_csv(output_file, index=False)

# Start using the ready file
original_df = pd.read_csv(output_file)
# Get the list of pixel column names (e.g., 'pix_1', 'pix_2', ..., 'pix_n')
pixel_columns = [col for col in original_df.columns if col.startswith('pix_')]

# Iterate through each pixel column
for i, pixel_column in enumerate(pixel_columns):
    # Create a new DataFrame with 'latitude', 'longitude', and the current pixel column
    new_df = original_df[['latitude', 'longitude', pixel_column]]
    # Create a new CSV file for each pixel column
    filename = f'{pixel_column}.csv'
    new_df.to_csv(filename, index=False)
    df = pd.read_csv(filename)
    new_coordinates(df, 2.5, i + 1)
    pivoting_orignal_files(df, i + 1)
    csv_file = f'destination_{i + 1}.csv'
    input_extract_index_header = f'pix_{i + 1}_coords.csv'
    extract_index_header(input_extract_index_header, i + 1)
    input_lat = f'unique_latitudes{i + 1}.csv'
    input_long = f'unique_longitudes{i + 1}.csv'
    with open(input_lat, 'r') as csvfile:
        reader = csv.reader(csvfile)
        data_lat = list(reader)
    lat_len = len(data_lat)
    with open(input_long, 'r') as csvfile:
        reader = csv.reader(csvfile)
        data_long = list(reader)
    long_len = len(data_long)
    BilinInterpolated_PixValue(csv_file, lat_len - 1, long_len - 1, i + 1)
    input_outputi = f'output{i + 1}.csv'
    leave_row_column(input_outputi, i + 1)
    concatinate_index_header(input_lat, input_long, input_outputi, i + 1)
    reverse_pivoting(input_outputi, i + 1)
    input_final = pd.read_csv(f'output_final_{i + 1}.csv')
    # clustering(area_name, input_final, i + 1)
try:
    n = len(pixel_columns)
    combine_output_final(n + 2,file_name)
except Exception as e:
    print(f"Error in combine_output_final: {e}")

try:
    source_csv = f'{input_file}'
    target_csv = f'aa_{file_name}.csv'
    preparing_aa(source_csv, target_csv)
except Exception as e:
    print(f"Error in preparing_aa: {e}")

try:
    # Here you will get the final file of the new coord and new values
    aa_file = f'aa_{file_name}.csv'

    output_file = f'2_Area_{file_name}_change.csv'
    change_detection(aa_file, output_file)
except Exception as e:
    print(f"Error in change_detection: {e}")

try:
    print("The Initial Area: ")
    area_calc(input_file, 10)
except Exception as e:
    print(f"Error in area_calc for initial area: {e}")

try:
    # Filtering according to the date
    inp = output_file
    date_filter(inp, area_name, file_name,inp_date)
except Exception as e:
    print(f"Error in date_filter: {e}")

try:
    print("The Change Detected Area: ")
    area_calc(f"1_Area_{file_name}_detected_data.csv", 5)
    if area_calc(f"1_Area_{file_name}_detected_data.csv", 5) < 0.001:
        print("No change")
        delete_un()
except Exception as e:
    print(f"Error in area_calc for change detected area: {e}")

try:
    # Clustering for cleaned_data
    input_data = f'1_Area_{file_name}_detected_data.csv'
    clustering(input_data,file_name)
except Exception as e:
    print(f"Error in clustering: {e}")

try:
    # Preparing cleaned data for 1 cluster - 1 date
    input_clean = f'cleaned_data_{file_name}.csv'
    cluster_date(input_clean,file_name)
except Exception as e:
    print(f"Error in cluster_date: {e}")

# creating a csvs with convex hull
# Convex_hull(input_clean)


# finding the grid number
pattern = r'_\d+'
match = re.search(pattern, input_file)
number = match.group()[1:]
# number = 10


# Call the function with your input file
input = f'cleaned_data_{file_name}.csv'
create_convex_hull_shapefile(input,number,file_name)

# # Opening Google earth and ploting the kml
# csv_to_kml('1_Area_detected_data.csv', '3output.kml')
# google_earth_path = r"C:\Program Files\Google\Google Earth Pro\client\googleearth.exe"
# kml_file = r"C:\Users\pdjns\OneDrive\Desktop\App.py\convex_hulls.shp"
# try:
#     subprocess.Popen([google_earth_path, kml_file])
# except Exception as e:
#     print("Error:", str(e))

# Deleting the files which got created in process



delete_un()
# Time calculation
end1 = time.time()
runtime = end1 - start1
print(f'Total runtime - {runtime}')
