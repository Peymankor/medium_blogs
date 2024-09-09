library(gt)

# Define the start and end dates for the data range
start_date <- "2010-06-07"
end_date <- "2010-06-14"

sp500 |> 
  dplyr::filter(date >= start_date & date <= end_date) |>
  dplyr::select(-adj_close) |>
  gt() |>
  tab_header(
    title = "S&P 500",
    subtitle = glue::glue("{start_date} to {end_date}")
  ) |>
  fmt_date(columns = date,
           date_style = "wd_m_day_year")

##################################################

Name <- c("Safety, 0-10 scale","NPV, USD million",
          "IRR, %", "Reserve added, million STB",
          "First year production, million STB",
          "Risk, probable NPV<0")

Name

Swing_Rank = c(3,1, 4, 5, 2, 6)

Weights_Abs = c(60, 100, 40, 30, 90, 20)

Weights_Norm = c(0.18, 0.29, 0.12, 0.09, 0.26, 0.06)

A <- c( 40, 70, 100, 90, 60, 40)

B <- c( 10, 0, 40, 80, 100, 80)

C <- c(0, 100, 90, 100, 50, 0)

D <- c(100, 30, 0, 70, 0, 100)

E <- c(80, 60, 30, 0, 40, 90)

newData <- data.frame(Name, Swing_Rank, Weights_Abs, Weights_Norm, A, 
                      B, C, D, E)

newData

newData |>
  gt() |>
  cols_label(Swing_Rank="Swing Rank")|>
  tab_spanner(
    label= md("**Alternatives**"),
    columns = 5:9
  ) |>
  tab_spanner(
    label= md("**Weights**"),
    columns = 3:4
  ) |>
  tab_spanner(
    label = md("**Objectives**"),
    columns = 1:4
  ) |>
  cols_label(Weights_Abs="Abs")|>
  cols_label(Weights_Norm="Norm.") 


install.packages("knitr")



############################################################################
#############################################################################



# Sample dataframe
df <- data.frame(
  A = c(1, 2, 3),
  B = c(4, 5, 6),
  C = c(7, 8, 9)
)

# Replace cells in column 'B' greater than 5 with 10
df$B <- replace(df$B, df$B > 5, 10)

# Print the modified dataframe
print(df)


#################################



# Sample dataframe
df <- data.frame(
  A = c(1, 2, 3),
  B = c(4, 5, 6),
  C = c(7, 8, 9)
)

# Row index and column name
row_index <- 2  # Index of the row you want to modify
column_name <- "B"  # Name of the column you want to modify

# New value to replace
new_value <- 10

# Replace cell in the specified row and column
df[row_index, column_name] <- new_value

# Print the modified dataframe
print(df)



####################


# Sample dataframe
df <- data.frame(
  A = c(1, 2, 3),
  B = c("x", "y", "z"),
  C = c(7, 8, 9)
)

# Key-value pairs for replacement
replacement_dict <- c("x" = 10, "y" = 20, "z" = 30)

# Loop through the dataframe and replace values
for (col_name in names(df)) {
  for (row_index in 1:nrow(df)) {
    current_value <- df[row_index, col_name]
    if (current_value %in% names(replacement_dict)) {
      df[row_index, col_name] <- replacement_dict[current_value]
    }
  }
}

# Print the modified dataframe
print(df)









# Sample data

linearconvertor <- function(scores, 
                            ) {
  map_range <- function(x, from_min, from_max, to_min, to_max) {
    ((x - from_min) / (from_max - from_min)) * (to_max - to_min) + to_min
  }
  
  scaled_values <- map_range(values, -2, 1, 0, 100)
  
}   
    return (scaled_values)

    
    
    values <- c(-2, -1, 0, 1)

# Function to map values from one range to another
map_range <- function(x, from_min, from_max, to_min, to_max) {
  ((x - from_min) / (from_max - from_min)) * (to_max - to_min) + to_min
}

# Apply the mapping function to scale values from -2 to 1 to a range from 0 to 100
scaled_values <- map_range(values, -2, 1, 0, 100)

scaled_values


linearconvertor <- function(scores, from_min, from_max, to_min, to_max) {
  
  map_range <- function(x, from_min, from_max, to_min, to_max) {
    ((x - from_min) / (from_max - from_min)) * (to_max - to_min) + to_min
  }
  
  scaled_values <- map_range(scores, -2, 1, 0, 100)
  
  return(scaled_values)
} 

linearconvertor(scores = c(-2,-1,0,1), 
                from_min = 0,
                from_max = 1,
                to_min = 0,
                to_max = 100)




exponential_convertor <- function(scores, from_min, from_max, to_min, to_max) {
  
  map_range <- function(x, from_min, from_max, to_min, to_max) {
    exp_value <- exp((x - from_min) / (from_max - from_min))
    scaled_value <- (exp_value - exp(from_min / (from_max - from_min))) / (exp(to_max / (from_max - from_min)) - exp(from_min / (from_max - from_min))) * (to_max - to_min) + to_min
    return(scaled_value)
  }
  
  scaled_values <- map_range(scores, from_min, from_max, to_min, to_max)
  
  return(scaled_values)
} 


exponential_convertor(scores = values,
                      from_min = -2,
                      from_max = 2,
                      to_min = 0,
                      to_max = 100)

# Example usage:
values <- c(-2, -1, 0, 0.5, 1)
scaled_values <- exponential_convertor(values, -2, 1, 0, 100)
scaled_values

exponential_convertor <- function(scores, from_min,
                                  from_max,
                                  to_min,
                                  to_max) {
  
  map_range <- function(x, from_min, from_max, to_min, to_max) {
    exp_value <- exp((x - from_min) / (from_max - from_min))
    scaled_value <- (exp_value - exp(from_min / (from_max - from_min))) / (exp(to_max / (from_max - from_min)) - exp(from_min / (from_max - from_min))) * (to_max - to_min) + to_min
    return(scaled_value)
  }
  
  scaled_values <- map_range(scores, min(scores), max(scores), 0, 100)
  
  return(scaled_values)
} 

# Example usage:
values <- c(-2, -1, 0, 0.5, 1)
scaled_values <- exponential_convertor(values,
                                       from_min = -2,
                                       from_max = 1,
                                       to_min = 0,
                                       to_max = 100)
scaled_values





exponential_converter <- function(scores, from_min, from_max, to_min, to_max) {
  
  map_range <- function(x, from_min, from_max, to_min, to_max) {
    exp_value <- exp((x - from_min) / (from_max - from_min))
    mapped_value <- ((exp_value - exp(from_min)) / (exp(from_max) - exp(from_min))) * (to_max - to_min) + to_min
    return(mapped_value)
  }
  
  scaled_values <- map_range(scores, from_min, from_max, to_min, to_max)
  
  return(scaled_values)
}



exponential_converter <- function(scores, from_min, from_max, to_min, to_max) {
  
  map_range <- function(x, from_min, from_max, to_min, to_max) {
    exp_value <- exp((x - from_min) / (from_max - from_min))
    mapped_value <- ((exp_value - exp(from_min / (from_max - from_min))) / (exp(from_max / (from_max - from_min)) - exp(from_min / (from_max - from_min)))) * (to_max - to_min) + to_min
    return(mapped_value)
  }
  
  scaled_values <- map_range(scores, from_min, from_max, to_min, to_max)
  
  return(scaled_values)
}

# Test the function
scores <- c(-2, -1, 0, 1, 2)
exponential_converter(scores, -2, 2, 0, 100)


exponential_converter(scores = c(-2,-1,0,1,2),
                      from_min = -2,
                      from_max = 2,
                      to_min = 0,
                      to_max = 100)



exponential_converter <- function(scores, from_min, from_max, to_min, to_max) {
  
  map_range <- function(x, from_min, from_max, to_min, to_max) {
    exp_value <- exp((x - from_min) / (from_max - from_min) * log(100) / log(exp(1)))
    mapped_value <- ((exp_value - exp(from_min / (from_max - from_min) * log(100) / log(exp(1)))) / (exp(from_max / (from_max - from_min) * log(100) / log(exp(1))) - exp(from_min / (from_max - from_min) * log(100) / log(exp(1))))) * (to_max - to_min) + to_min
    return(mapped_value)
  }
  
  scaled_values <- map_range(scores, from_min, from_max, to_min, to_max)
  
  return(scaled_values)
}

# Test the function
scores <- c(-2, -1, 0, 1, 2)
exponential_converter(scores, -2, 2, 0, 100)


exponential_converter <- function(scores, from_min, from_max, to_min, to_max) {
  exp_min <- exp(from_min)  # Exponential value for minimum score
  exp_max <- exp(from_max)  # Exponential value for maximum score
  
  # Calculate the mapped values using exponential function
  mapped_values <- ((exp(scores) - exp_min) / (exp_max - exp_min)) * (to_max - to_min) + to_min
  
  return(mapped_values)
}

# Test the function
scores <- c(-2, -1, 0, 1, 2)
scores <- seq(-2,2, by=0.1)
yy <- exponential_converter(scores, -2, 2, 0, 100)

plot(scores,yy)

exponential_converter <- function(scores, from_min, from_max, to_min, to_max) {
  # Scale scores to a range from 0 to 100
  scaled_scores <- ((scores - min(scores)) / (max(scores) - min(scores))) * 100
  
  # Find fixed parameter using scaled score
  fixed_parameter <- -log(scaled_scores) / max(scores)
  
  # Calculate mapped values using modified exponential function
  mapped_values <- ((exp(-scores * fixed_parameter) - exp(-from_min * fixed_parameter)) / (exp(-from_max * fixed_parameter) - exp(-from_min * fixed_parameter))) * (to_max - to_min) + to_min
  
  return(mapped_values)
}


exponential_converter(scores = seq(-1.9,2, by=0.1),
                      from_min = -2,
                      from_max = 2,
                      to_min = 0,
                      to_max = 100)


exp_decay <- function(x, rho) {
  return(exp(x / rho))
}

xx <- seq(-2,2, by=0.1)
plot(xx, exp_decay(xx,13))

rescaled_exp_decay <- function(x) {
  # Solve for rho
  rho <- -2 / log(100)
  
  # Calculate the rescaled values
  rescaled_values <- (1 - exp(-x / rho)) * 100
  
  return(rescaled_values)
}


xx <- c(-2,-1,0,1,1,2)
rescaled_exp_decay(xx)


rescaled_exp_decay <- function(x) {
  # Solve for rho
  rho <- -2 / log(100/0.01)
  
  # Calculate the rescaled values
  rescaled_values <- (1 - exp(-x / rho)) * 100
  
  return(rescaled_values)
}

# Example usage
x_values <- c(-2, 2)
rescaled_values <- rescaled_exp_decay(x_values)
print(rescaled_values)

exp_decay_rescaled <- function(x) {
  # Solve for rho
  rho <- -2 / log(100/0.01)
  
  # Calculate the rescaled values
  rescaled_values <- (1 - exp(-x / rho)) * 100
  
  return(rescaled_values)
}


plot(seq(-2,2,0.1), exp_decay_rescaled(seq(-2,2,0.1)))


exp_decay <- function(x, rho) {
  scale_factor <- 100 / (exp(2 / rho) - exp(-2 / rho))
  offset <- -scale_factor * exp(-2 / rho)
  return(-scale_factor * exp(x / rho) + offset)
}

exp_decay(2,100)

xx <- seq(-2,2,0.1)
xx
plot(xx, exp_decay(xx,1))



# Sample dataframe
df <- data.frame(
  ID = c(1, 2, 3, 4),
  Name = c("John", "Alice", "Bob", "Emily"),
  Age = c(25, 30, 35, 40)
)

# Get the number of rows in the dataframe
num_rows <- nrow(df)

# Iterate through each row using a for loop
for (i in 1:num_rows) {
  # Access values in each row using index 'i'
  row_values <- df[i, ]
  print(row_values)
}



scale_values <- function(x, min_val, max_val) {
  # Compute the slope and intercept of the linear transformation
  slope <- 100 / (max_val - min_val)
  intercept <- -min_val * slope
  
  # Apply the linear transformation
  scaled_values <- slope * x + intercept
  
  # Ensure values are within the range [0, 100]
  scaled_values <- pmin(pmax(scaled_values, 0), 100)
  
  return(scaled_values)
}

# Example usage:
values <- c(2800, 3000, 3200, 3400)
scaled <- scale_values(values, 2800, 3400)
print(scaled)



replace_row_with_scaled <- function(df, row_index, start_col, end_col, digits = 1) {
  row_data <- as.numeric(unlist(df[row_index, start_col:end_col]))
  row_scaled <- scale_values(row_data, min_val = min(row_data), max_val = max(row_data))
  df[row_index, start_col:end_col] <- round(row_scaled, digits = digits)
  return(df)
}

# Example usage:
# Assuming you have a matrix consmatrix_copy and a function scale_values defined:

# consmatrix_copy <- data.frame(...) # Your data frame
# scale_values <- function(x, min_val, max_val) { return((x - min_val) / (max_val - min_val)) } # Your scale_values function

# Replace row 6 with scaled values from columns 2 to 6
# consmatrix_copy <- replace_row_with_scaled(consmatrix_copy, 6, 2, 6)


replace_rows_with_scaled <- function(df, start_row, end_row, start_col, end_col, digits = 1) {
  for (i in start_row:end_row) {
    row_data <- as.numeric(unlist(df[i, start_col:end_col]))
    row_scaled <- scale_values(row_data, min_val = min(row_data), max_val = max(row_data))
    df[i, start_col:end_col] <- round(row_scaled, digits = digits)
  }
  return(df)
}



# Example usage:
# Assuming you have a matrix consmatrix_copy and a function scale_values defined:

# consmatrix_copy <- data.frame(...) # Your data frame
# scale_values <- function(x, min_val, max_val) { return((x - min_val) / (max_val - min_val)) } # Your scale_values function

# Replace rows from 1 to 6 with scaled values from columns 2 to 6
# consmatrix_copy <- replace_rows_with_scaled(consmatrix_copy, 1, 6, 2, 6)


# Define the function
f <- function(x, a, b, c, d) {
  a * exp(-b * x) + c + d / (1 + exp(x))
}

# Set parameters (adjust as needed)
a <- 2  # scaling factor
b <- 0.5  # controls initial rate of change
c <- 1  # vertical shift
d <- 5  # controls the upper limit (adjust as needed)

# Create sequence of x values
x <- seq(from = 0, to = 5, length = 100)

# Calculate y values
y <- f(x, a, b, c, d)

# Plot the curve
plot(x, y, type = "l", 
     xlab = "Input (x)", 
     ylab = "Output (y)",
     main = "Exponential Curve with Slower Change Initially, Increasing and Approaching d")

# Add reference lines (optional)
abline(v = 0, col = "gray", lty = 2)
abline(h = c, col = "gray", lty = 2)
abline(h = c + d, col = "gray", lty = 2)


# Define the function
f <- function(x, a, b, c, d) {
  a * (1 - exp(-b * x)) + c + d / (1 + exp(-x))
}

# Set parameters (adjust as needed)
a <- 2  # scaling factor
b <- 0.5  # controls initial rate of change
c <- 1  # vertical shift
d <- 5  # controls the upper limit (adjust as needed)

# Create sequence of x values
x <- seq(from = 0, to = 5, length = 100)

# Calculate y values
y <- f(x, a, b, c, d)

# Plot the curve
plot(x, y, type = "l", 
     xlab = "Input (x)", 
     ylab = "Output (y)",
     main = "Exponential Curve with Slower Change Initially, Increasing and Approaching d")

# Add reference lines (optional)
abline(v = 0, col = "gray", lty = 2)
abline(h = c, col = "gray", lty = 2)
abline(h = c + d, col = "gray", lty = 2)

# You can adjust the horizontal range to visualize the behavior better

# You can adjust the horizontal range to visualize the behavior better


# Define the logistic function
f <- function(x, a, b, c, d) {
  -exp(-a * (x - b))
}

rescale_function <- function(x, a, b, min_val, max_val) {
  # Calculate the original function values
  y <- -exp(-a * (x - b))
  
  # Rescale the values to the desired range (0 to 100)
  #rescaled_y <- (y - min(y)) * ((max_val - min_val)) / (max(y) - min(y)) 
  
  return(y)
}

# Set parameters (adjust as needed)
a <- 1  # controls the steepness of the curve
b <- 0  # controls the horizontal shift
c <- 100  # vertical shift
d <- 5  # controls the upper limit (adjust as needed)

# Create sequence of x values
x <- seq(from = 0, to = 4, length = 100)

# Calculate y values

y <- f(x, a, b, c, d)

y
# Plot the curve
plot(x, y, type = "l", 
     xlab = "Input (x)", 
     ylab = "Output (y)",
     main = "Logistic Curve with Slower Change Initially, Increasing and Approaching d")

# Add reference lines (optional)
abline(v = 0, col = "gray", lty = 2)
abline(h = c, col = "gray", lty = 2)
abline(h = c + d, col = "gray", lty = 2)

# You can adjust the horizontal range to visualize the behavior better


rescale_function <- function(x, a, b) {
  # Calculate the original function values
  y <- -exp(-a * (x - b))
  
  # Rescale the values using linear transformation
  m = (100 - 0) / (4 - 0)  # Slope for linear transformation
  c = 0 - m * 0  # y-intercept for linear transformation
  rescaled_y = m * y + c
  
  return(rescaled_y)
}

# Define parameters
a <- 2  # Controls the shape of the curve
b <- 0.5  # Controls the horizontal shift

# Generate x values
x <- seq(from = 0, to = 4, length = 100)

# Calculate the rescaled function values
rescaled_y <- rescale_function(x, a, b)

# Print the first 10 rescaled values (optional)
rescaled_y


# Define the function with scaling and shifting
f <- function(x, a, b) {
  # Scale the output to the range (0, 100)
  100 * (-exp(-a * (x - b))) / (1 - exp(-a * 4))
}

# Set parameters (adjust a as needed)
a <- 2  # Controls the shape of the curve
b <- 2  # Adjusted to achieve y = 0 at x = 0

# Generate x values
x <- seq(from = 0, to = 4, length = 100)

# Calculate y values
y <- f(x, a, b)

# Plot the curve
plot(x, y, type = "l", xlab = "x", ylab = "y")

# Add reference lines (optional)
abline(v = 0, col = "gray", lty = 2)
abline(h = 0, col = "gray", lty = 2)
abline(h = 100, col = "gray", lty = 2)


r_function <- function(x, a, b) {
  # Calculate the original function values
  y <- -exp(-a * (x - b))
  
  # Rescale y to be between 0 and 100
  scaled_y <- ((y - min(y)) / (max(y) - min(y))) * 100
  
  return(scaled_y)
}


x <- seq(from=-1, to=3, length=100)


yy <- r_function(x, 1, 2)
plot(x, yy, type = "l", xlab = "x", ylab = "y")


# Define a list of numbers
list_of_numbers <- c(1, 2, 3, 4, 5)

# Check if the number 3 is in the list of numbers
if (3 %in% list_of_numbers) {
  print("3 is in the list of numbers")
} else {
  print("3 is not in the list of numbers")
}


replace_rows_with_scaled <- function(df, start_row, end_row, start_col, end_col, digits = 1, list_oflinear) {
  
  df_copy <- data.frame(df) 
  
  for (i in start_row:end_row) {
    row_data <- as.numeric(unlist(df[i, start_col:end_col]))
    
    if (i %in% list_oflinear) {
      row_scaled <- scale_values_linear(row_data, min_val = min(row_data), max_val = max(row_data))
    } else {
      row_scaled <- exp_function(row_data, a = 1, b = 0)
    }
    
    df_copy[i, start_col:end_col] <- round(row_scaled, digits = digits)
  }
  
  return(df_copy)
}
