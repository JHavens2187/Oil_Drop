import pandas as pd
import numpy as np
import os
from scipy.stats import zscore
import matplotlib.pyplot as plt

real_charge = 1.60217663e-19

#========================= data ==============================#
# Data from the provided image
millikan_data = pd.DataFrame({
    'G_seconds': [120.8, 121.0, 121.2, 120.1, 120.2, 119.8, 120.1, 120.2, 119.9],
    'F_seconds': [26.2, 11.9, 16.5, 16.3, 26.4, 67.4, 26.6, 16.6, 67.8],
    'n': [2, 4, 3, 3, 2, 1, 2, 3, 1],
    'en_x10_10': [10.98, 21.98, 16.41, 16.41, 10.98, 5.495, 10.98, 16.41, 5.495],
    'e1_x10_10': [5.490, 5.495, 5.470, 5.470, 5.495, 5.495, 5.490, 5.470, 5.495],
    'd': 1.303,
    'T': 24.6,
    'density': 0.9041,
    'V': 9150,
    'v1': 0.01085
})

# Data from the provided image
cornell_data = pd.DataFrame({
    'voltage_V': [3.060, 4.080, 3.020, 3.800, 3.400, 3.400, 3.400, 3.68, 3.68, 3.68, 4.987, 4.987, 3.122, 3.122, 3.122,
                  3.780, 3.780, 3.298, 3.782, 3.782, 3.782, 3.700, 3.700, 3.782, 3.784, 3.784, 3.782, 3.782, 3.459],
    'neutral_fall_seconds': [8.94, 10.26, 10.57, 10.39, 11.62, 11.48, 11.41, 12.04, 12.12, 12.21, 12.67, 12.60, 13.04,
                             12.98, 13.12, 13.87, 13.61, 15.78, 16.72, 16.82, 16.51, 17.72, 17.76, 18.12, 18.78, 19.04,
                             19.72, 19.67, 20.23],
    'charged_rise_seconds': [58.94, 65.92, 7.61, 19.07, 12.30, 12.16, 12.34, 30.23, 30.92, 29.65, 6.89, 6.95, 39.90,
                             40.18, 40.06, 10.30, 10.21, 57.41, 7.78, 7.61, 7.35, 25.49, 24.26, 23.10, 10.04, 9.83,
                             6.35, 6.33, 21.63],
    'r_prime_cm': [1.90e-04, 1.78e-04, 1.75e-04, 1.77e-04, 1.67e-04, 1.68e-04, 1.69e-04, 1.64e-04, 1.64e-04, 1.63e-04,
                   1.60e-04, 1.60e-04, 1.58e-04, 1.58e-04, 1.57e-04, 1.53e-04, 1.54e-04, 1.43e-04, 1.39e-04, 1.39e-04,
                   1.40e-04, 1.35e-04, 1.35e-04, 1.34e-04, 1.31e-04, 1.30e-04, 1.28e-04, 1.28e-04, 1.27e-04],
    'q_prime_esu': [2.63e-09, 1.61e-09, 4.29e-09, 2.26e-09, 2.69e-09, 2.74e-09, 2.74e-09, 1.70e-09, 1.67e-09, 1.68e-09,
                    2.35e-09, 2.35e-09, 1.68e-09, 1.69e-09, 1.67e-09, 2.24e-09, 2.29e-09, 1.15e-09, 2.27e-09, 2.29e-09,
                    2.39e-09, 1.15e-09, 1.17e-09, 1.14e-09, 1.74e-09, 1.74e-09, 2.31e-09, 2.32e-09, 1.15e-09],
    'n_prime': [5.47, 3.35, 8.94, 4.72, 5.61, 5.71, 5.71, 3.53, 3.48, 3.49, 4.90, 4.90, 3.51, 3.52, 3.48, 4.67, 4.78,
                2.40, 4.73, 4.78, 4.97, 2.39, 2.43, 2.38, 3.62, 3.63, 4.82, 4.84, 2.39],
    'n': [5, 3, 8, 4, 5, 5, 5, 3, 3, 3, 4, 4, 3, 3, 3, 4, 4, 2, 4, 4, 4, 2, 2, 2, 3, 3, 4, 4, 2],
    'e_2_3': [3.131e-13, 3.173e-13, 3.176e-13, 3.291e-13, 3.184e-13, 3.222e-13, 3.220e-13, 3.288e-13, 3.257e-13,
              3.263e-13, 3.377e-13, 3.375e-13, 3.272e-13, 3.280e-13, 3.253e-13, 3.269e-13, 3.318e-13, 3.326e-13,
              3.298e-13, 3.321e-13, 3.408e-13, 3.317e-13, 3.357e-13, 3.308e-13, 3.343e-13, 3.348e-13, 3.337e-13,
              3.346e-13, 3.319e-13],
    '1_r_prime': [5252, 5627, 5711, 5662, 5988, 5952, 5933, 6095, 6115, 6138, 6253, 6235, 6343, 6329, 6363, 6542, 6480,
                  6978, 7183, 7204, 7137, 7394, 7403, 7477, 7612, 7665, 7800, 7791, 7901]
})

# Data from the provided image (KSU data)
ksu_data = pd.DataFrame({
    'droplet': [1, 2, 3, 4, 5, 6],
    'v_plus_cms': [0.3, 0.3, 0.6, 0.6, 0.98, 0.03],
    'v_minus_cms': [0.6, 0.89, 0.45, 0.23, 0.5, 0.5],
    'v_down_cms': [0.03, 0.8, 0.56, 0.34, 0.55, 0.23],
    'e_up_10_19_Coul': [1.54, 1.67, 1.5, 1.7, 1.4, 1.4],
    'e_down_10_19_Coul': [1.4, 1.6, 1.5, 1.7, 1.4, 2.0],
    'e_up_uncertainty_10_19_Coul': [0.2, 0.2, 0.24, 0.28, 0.23, 0.29],
    'e_down_uncertainty_10_19_Coul': [0.25, 0.27, 0.26, 0.28, 0.25, 0.40]
})

Erp_data = pd.DataFrame({
    'Q_C_Experimental': [1.63612e-19, 1.57929e-19, 1.58111e-19, 1.45695e-19, 1.36383e-19, 2.00193e-19, 1.88168e-19,
                         1.89162e-19, 1.89845e-19, 2.01452e-19, 1.76861e-19, 1.68751e-19, 1.30269e-19, 1.57835e-19,
                         1.58080e-19, 1.77047e-19, 1.58461e-19, 1.73115e-19, 1.71089e-19, 1.23468e-19, 1.32611e-19,
                         1.31779e-19, 1.25432e-19, 1.16653e-19, 1.20306e-19],
    'Q_T_Theoretical': [1.60217733e-19] * 25,  # Repeated theoretical value
    'Error_Percent': [2.11, 1.43, 1.31, 9.06, 14.87, 24.95, 17.44, 18.06, 18.49, 25.73, 10.38, 5.32, 18.69, 1.49, 1.33,
                      10.50, 1.10, 8.05, 6.79, 22.94, 17.23, 17.74, 21.69, 27.26, 24.91]
})


#========================= functions ==============================#

# Function to remove outliers
def remove_outliers(df, cols):
    if len(df) <= 1:
        return df  # Ensure at least one row remains

    filtered_df = df.copy()
    for col in cols:
        mu, sigma = filtered_df[col].mean(), filtered_df[col].std()
        filtered_df = filtered_df[np.abs(filtered_df[col] - mu) <= 2 * sigma]
    return filtered_df if not filtered_df.empty else df.iloc[[0]]


# Load data from CSV files
data_files = [os.path.join("data", f) for f in os.listdir("data") if f.endswith(".csv")]
data = [remove_outliers(pd.read_csv(f), ["neutral_fall_seconds", "charged_rise_seconds"]) for f in data_files]


# Compute charges
def compute_charge(df):
    d_lines = (0.5e-3, 0.1e-4)
    vf, vr = d_lines[0] / df["neutral_fall_seconds"].mean(), d_lines[0] / df["charged_rise_seconds"].mean()
    voltage_error = round(df["voltage_V"].mean() * 0.0009 + 0.2, 1)
    V, d, E = (df["voltage_V"].mean(), voltage_error), df["d"].mean(), df["voltage_V"].mean() / df["d"].mean()
    eta, rho, b, p, g = df["eta"].mean(), df["rho"].mean(), df["b"].mean(), df["p"].mean(), df["g"].mean()
    a = np.sqrt((b / (2 * p)) ** 2 + (9 * eta * vf) / (2 * g * rho)) - b / (2 * p)
    m = (4 / 3) * np.pi * a ** 3 * rho
    return m * g * (vf + vr) / (E * vf)


def calculate_millikan_charge(millikan_data):
    """
    Calculate the charge values based on Millikan's data.
    """
    e1 = millikan_data['e1_x10_10'] * 1e-10  # Convert to Coulombs
    en = millikan_data['en_x10_10']  # Convert to Coulombs
    d = millikan_data['d']  # Diameter of the droplet (m)
    T = millikan_data['T']  # Temperature in Celsius
    density = millikan_data['density']  # Density of the oil droplet (kg/m^3)
    V = millikan_data['V']  # Voltage (V)

    # Using the equation for the charge calculation in Millikan's experiment
    q_values = en
    return q_values * real_charge


def calculate_cornell_charge(cornell_data):
    """
    Calculate the charge values based on Cornell data.
    """
    voltage = cornell_data['voltage_V']
    neutral_fall_seconds = cornell_data['neutral_fall_seconds']
    charged_rise_seconds = cornell_data['charged_rise_seconds']
    r_prime_cm = cornell_data['r_prime_cm']
    q_prime_esu = cornell_data['q_prime_esu']

    # Charge in Coulombs from the experimental data
    q_values_coulombs = q_prime_esu * 3.3356e-10  # Convert from ESU to Coulombs
    return q_values_coulombs


def calculate_ksu_charge(ksu_data):
    """
    Calculate the charge values based on KSU's data.
    """
    e_up = ksu_data['e_up_10_19_Coul']  # Electric field (Coulombs)
    e_down = ksu_data['e_down_10_19_Coul']

    # Calculating the average charge per droplet
    q_values = (e_up + e_down) / 2  # Averaging the up and down values

    return q_values * real_charge


def calculate_erp_charge(Erp_data):
    """
    Calculate the charge values based on Erp's data.
    """
    Q_C_Experimental = Erp_data['Q_C_Experimental']
    Q_T_Theoretical = Erp_data['Q_T_Theoretical']

    # Theoretical vs experimental charge values comparison (in Coulombs)
    charge_values = Q_C_Experimental
    return charge_values


# compute charges for all data
charges = [compute_charge(df) for df in data]
milikancharges = calculate_millikan_charge(millikan_data)
cornellcharges = calculate_cornell_charge(cornell_data)
ksucharges = calculate_ksu_charge(ksu_data)
erpcharges = calculate_erp_charge(Erp_data)
print(charges[0], milikancharges[0], cornellcharges[0], ksucharges[0], erpcharges[0])

# Normalize charges
normalized_charge = np.array(charges) /real_charge
normalized_milikan_charge = np.array(milikancharges) / real_charge
normalized_cornell_charge = np.array(cornellcharges) / real_charge
normalized_ksu_charge = np.array(ksucharges) / real_charge
normalized_erp_charge = np.array(erpcharges) / real_charge
print(normalized_charge[0], normalized_milikan_charge[0], normalized_cornell_charge[0], normalized_ksu_charge[0], normalized_erp_charge[0])

#========================= errors ==============================#
num_measurements = [len(df) for df in data]

# Calculate the statistical and systematic uncertainties for my data
systematic_uncertainties = np.std(charges)
statistical_uncertainties = real_charge / np.sqrt(num_measurements)

# Calculate total uncertainty
total_uncertainty = np.sqrt(systematic_uncertainties ** 2 + statistical_uncertainties ** 2)

# Z-scores for the measured data
z_scores = (charges / normalized_charge - real_charge) / np.sqrt(
    statistical_uncertainties ** 2 + systematic_uncertainties ** 2)
chi_squared = np.sum(z_scores ** 2)

# Millikan data uncertainties
num_measurements_m = len(millikan_data)
systematic_uncertainties_m = np.std(milikancharges)
statistical_uncertainties_m = real_charge / np.sqrt(num_measurements_m)

# Calculate total uncertainty for Millikan data
total_uncertainty_m = np.sqrt(systematic_uncertainties_m ** 2 + statistical_uncertainties_m ** 2)

z_scores_m = (milikancharges / normalized_milikan_charge - real_charge) / np.sqrt(
    statistical_uncertainties_m ** 2 + systematic_uncertainties_m ** 2)
chi_squared_m = np.sum(z_scores_m ** 2)

# error analysis for the cornell data
systematic_uncertainties_c = np.std(cornellcharges)
statistical_uncertainties_c = real_charge / np.sqrt(len(cornell_data))

# Calculate total uncertainty for Cornell data
total_uncertainty_c = np.sqrt(systematic_uncertainties_c ** 2 + statistical_uncertainties_c ** 2)

z_scores_c = (cornellcharges / normalized_cornell_charge - real_charge) / np.sqrt(
    statistical_uncertainties_c ** 2 + systematic_uncertainties_c ** 2)
chi_squared_c = np.sum(z_scores_c ** 2)

# error "analysis" (theres no analysis, KSU provides their error)
total_uncertainty_k = (ksu_data['e_up_uncertainty_10_19_Coul'] + ksu_data['e_down_uncertainty_10_19_Coul']) / 2 * real_charge

z_scores_k = (ksucharges / normalized_ksu_charge - real_charge) / total_uncertainty_k
chi_squared_k = np.sum(z_scores_k ** 2)

# error analysis for the erp data
systematic_uncertainties_e = np.std(erpcharges)
statistical_uncertainties_e = real_charge / np.sqrt(len(Erp_data))

# Calculate total uncertainty for Erp data
error_percent = Erp_data['Error_Percent']
#total_uncertainty_e = error_percent * erpcharges / 10
total_uncertainty_e = np.sqrt(systematic_uncertainties_e ** 2 + error_percent ** 2) * real_charge
z_scores_e = (erpcharges / normalized_erp_charge - real_charge) / total_uncertainty_e
chi_squared_e = np.sum(z_scores_e ** 2)

# Generate droplet labels
droplet_labels = [chr(65 + i) for i in range(len(charges))]
droplet_labels_m = [chr(65 + i) for i in range(len(milikancharges))]
droplet_labels_c = [chr(65 + i) for i in range(len(cornellcharges))]
droplet_labels_k = [chr(65 + i) for i in range(len(ksucharges))]
droplet_labels_e = [chr(65 + i) for i in range(len(erpcharges))]

# Create results DataFrame
df_results = pd.DataFrame({
    "Droplet": droplet_labels,
    "Charge (C)": np.round(charges, 4),
    "# Charges": np.round(normalized_charge, 4),
    "Statistical Uncertainty": np.round(statistical_uncertainties, 4),
    "Systematic Uncertainty": np.round(systematic_uncertainties, 4),
    "Total Uncertainty": np.round(total_uncertainty, 4),
    "Z-Score": np.round(z_scores, 4)
})

mdf_results = pd.DataFrame({
    "Droplet": droplet_labels_m,
    "Charge (C)": np.round(milikancharges, 4),
    "# Charges": np.round(normalized_milikan_charge, 4),
    "Statistical Uncertainty": np.round(statistical_uncertainties_m, 4),
    "Systematic Uncertainty": np.round(systematic_uncertainties_m, 4),
    "Total Uncertainty": np.round(total_uncertainty_m, 4),
    "Z-Score": np.round(z_scores_m, 4)
})

cdf_results = pd.DataFrame({
    "Droplet": droplet_labels_c,
    "Charge (C)": np.round(cornellcharges, 4),
    "# Charges": np.round(normalized_cornell_charge, 4),
    "Statistical Uncertainty": np.round(statistical_uncertainties_c, 4),
    "Systematic Uncertainty": np.round(systematic_uncertainties_c, 4),
    "Total Uncertainty": np.round(total_uncertainty_c, 4),
    "Z-Score": np.round(z_scores_c, 4)
})

kdf_results = pd.DataFrame({
    "Droplet": droplet_labels_k,
    "Charge (C)": np.round(ksucharges, 4),
    "# Charges": np.round(normalized_ksu_charge, 4),
    "Total Uncertainty": np.round(total_uncertainty_k, 4),
    "Z-Score": np.round(z_scores_k, 4)
})

edf_results = pd.DataFrame({
    "Droplet": droplet_labels_e,
    "Charge (C)": np.round(erpcharges, 4),
    "# Charges": np.round(normalized_erp_charge, 4),
    "Statistical Uncertainty": np.round(statistical_uncertainties_e, 4),
    "Systematic Uncertainty": np.round(systematic_uncertainties_e, 4),
    "Total Uncertainty": np.round(total_uncertainty_e, 4),
    "Z-Score": np.round(z_scores_e, 4)
})

# Print results
'''print("My Data:")
print(df_results, f"\nΧ² = {chi_squared}")
print("\nMillikan Data:")
print(mdf_results, f"\nΧ²_m = {chi_squared_m}")
print("\nCornell Data:")
print(cdf_results, f"\nΧ²_c = {chi_squared_c}")
print("\nKSU Data:")
print(kdf_results, f"\nΧ²_k = {chi_squared_k}")
print("\nErp Data:")
print(edf_results, f"\nΧ²_e = {chi_squared_e}")'''

#========================= plotting ==============================#
plt.figure(figsize=(8, 6))
scale = min(charges) * 1e19
y_tick_max = 1.629e-18 * 1e19
y_max_scaled = y_tick_max / scale
x, y = np.linspace(min(normalized_charge), max(normalized_milikan_charge), 10), np.linspace(min(normalized_charge),
                                                                                            max(normalized_milikan_charge), 10)

# Scale error bars the same way as charges
charge_error_scaled = total_uncertainty / np.array(charges)[::-1]
milikancharge_error_scaled = total_uncertainty_m / np.array(milikancharges)[::-1]
cornellcharge_error_scaled = total_uncertainty_c
ksucharge_error_scaled = total_uncertainty_k / np.array(ksucharges)[::-1]
erpcharge_error_scaled = total_uncertainty_e / np.array(erpcharges)[::-1]

plt.errorbar(normalized_charge, charges / min(charges),
             yerr=charge_error_scaled, fmt='ro', label='Measured Charges', capsize=3)
plt.errorbar(normalized_milikan_charge, milikancharges / min(milikancharges) * min(normalized_milikan_charge),
             yerr=np.abs(milikancharge_error_scaled), fmt='mo', label='Millikan Data', capsize=3)
plt.errorbar(normalized_cornell_charge, cornellcharges / min(cornellcharges) * min(normalized_cornell_charge),
             yerr=cornellcharge_error_scaled, fmt='bo', label='Cornell Data', capsize=3)
plt.errorbar(normalized_ksu_charge, ksucharges / min(ksucharges),
             yerr=ksucharge_error_scaled, fmt='yo', label='KSU Data', capsize=3)
plt.errorbar(normalized_erp_charge, erpcharges / min(erpcharges),
             yerr=erpcharge_error_scaled, fmt='co', label='Erp Data', capsize=3)

# Add title and labels
plt.title("Electron Charge via Millikan Oil Drop Experiment", fontsize=14)
plt.xlabel("Normalized Charge", fontsize=12)
plt.ylabel("Measured Charge / Elementary Charge", fontsize=12)

# Add a legend with citations in AAS/APS format
plt.legend(
    title="Data Sources and Citations",
    loc='lower right',
    fontsize=10,
    labels=[
        "Saggers and Guardiola (2025)",
        "Millikan (1910)",
        "Arias (1997)",
        "Kansas State University",
        "Erp (2018)"
    ]
)

# Plotting the ideal y=x line
plt.plot(x, y, 'k--', label='y=x')
plt.grid()
plt.savefig("output_plot.png")
plt.show()

#========================= residuals ==============================#

datasets = [charges, milikancharges, cornellcharges, ksucharges, erpcharges]
x_values = [normalized_charge, normalized_milikan_charge, normalized_cornell_charge, normalized_ksu_charge, normalized_erp_charge]
dataerror = [total_uncertainty, total_uncertainty_m, total_uncertainty_c, total_uncertainty_k, total_uncertainty_e]
colors = ['r', 'm', 'b', 'y', 'c']
labels = [
    "Saggers and Guardiola (2025)",
    "Millikan (1910)",
    "Arias (1997)",
    "Kansas State University",
    "Erp (2018)"
]

# Create a figure for the residuals plot
plt.figure(figsize=(10, 6))

# Loop through each dataset and calculate residuals
for i, (data, error, x, color, label) in enumerate(zip(datasets, dataerror, x_values, colors, labels)):
    # Calculate residuals (deviation from y = x)
    print(f"now checking {label}")
    if np.log10(data[0]) <= -1:
        residuals = x - (np.array(data) / real_charge)
    else:
        residuals = x - np.array(data)

    # Scale errors (apply a simple reduction factor)
    scaled_errors = error  # Adjust this factor as needed

    # Plot the residuals with scaled error bars
    plt.errorbar(x, residuals / 10, yerr=scaled_errors, fmt='o', color=color, label=label, capsize=3)

# Add zero residual line
plt.axhline(y=0, color='black', linestyle='--')

# Add title and labels
plt.title('Residuals Plot for Electron Charge Measurement', fontsize=14)
plt.xlabel('Normalized Charges', fontsize=12)
plt.ylabel('Residuals (C)', fontsize=12)
plt.yscale('linear')  # Use linear y-axis scale

# Add grid for better readability
plt.grid(True)

# Add the legend with citations
plt.legend(
    title="Data Sources and Citations",
    loc='upper right',
    fontsize=10
)

# Show the plot
plt.savefig("residuals_plot.png")
plt.show()
