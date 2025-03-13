library(tidyverse)
library(AER)
library(knitr)
library(tidyr)
library(xtable)


market_product_data <- read.csv("market_demand_simulated_data.csv")


### Question i) Estimate multinomial logit demand using OLS ###

# Step 1: Calculate outside option share for each market
market_shares <- market_product_data %>%
  group_by(market) %>%
  mutate(
    market_total_share = sum(s), # sum total market share by market of all non-outside options
    outside_share = 1 - market_total_share
  ) %>%
  ungroup()


# Step 2: Calculate dependent variable (log share ratio)
ols_logit_data <- market_shares %>%
  mutate(ln_share_ratio = log(s / outside_share))

# Step 3: Run OLS regression with product characteristics
# Including both price (p) and product characteristics (x, w)
# Also including the satellite indicator as a product characteristic
# "wired" is omitted to avoid perfect multicollinearity
ols_logit_model <- lm(ln_share_ratio ~ p + x + w + satellite, data = ols_logit_data)

# View results
summary(ols_logit_model)

# The intercept on price is positive.
# This suggests that the conumer utility increases as price increases.
# This is counter to what economic theory would suggest.
# This may be because of endogenous prices biasing OLS.
# The coefficient on satellite is positive, suggesting slightly more consumer utility compared to wired.




#### Question ii) multinomial logit model of demand by two-stage least squares, instrumenting for prices ###

# Step 1: Calculate outside option share for each market
# (Same as before)
market_shares <- market_product_data %>%
  group_by(market) %>%
  mutate(
    market_total_share = sum(s),
    outside_share = 1 - market_total_share
  ) %>%
  ungroup()

# Step 2: Calculate dependent variable (log share ratio)
logit_data <- market_shares %>%
  mutate(ln_share_ratio = log(s / outside_share))

# Step 3: Implement 2SLS regression
# First stage: regress price (p) on instruments (x and w) and exogenous variables
# Second stage: use predicted prices in the main regression

# Using the ivreg function from AER package
# Formula syntax: dependent_var ~ exogenous_vars | instruments
# In this case:
# - ln_share_ratio is the dependent variable
# - satellite is exogenous
# - p is endogenous
# - x and w are instruments for p


# We assume "satellite" is exogenous - independent of unobserved quality factors 

iv_logit_model <- ivreg(ln_share_ratio ~ satellite + p | satellite + x + w, 
                        data = logit_data)

# View results
summary(iv_logit_model)

# The sign of price reverses compared to oLS.
# Now the price effect on utility is negative, which is what we would expect.
# Also, the coefficient on satellite is slightly negative, suggesting less consumer utility compare to wired.




### Question iii) Nested logit ###




# Step 1: Calculate market shares and nest shares
nested_data <- market_product_data %>%
  # Calculate total market shares by market
  group_by(market) %>%
  mutate(
    market_total_share = sum(s),
    outside_share = 1 - market_total_share
  ) %>%
  # Calculate within-nest shares
  group_by(market, satellite) %>%
  mutate(
    nest_total_share = sum(s),
    within_nest_share = s / nest_total_share
  ) %>%
  ungroup() %>%
  # Create the dependent variable
  mutate(
    ln_share_ratio = log(s / outside_share),
    ln_within_nest_share = log(within_nest_share)
  ) %>%
  # Create interaction terms for different nesting parameters
  mutate(
    ln_within_nest_share_satellite = ln_within_nest_share * satellite,
    ln_within_nest_share_wired = ln_within_nest_share * (1-satellite)
  )

# Step 2: Implement 2SLS regression with separate nesting parameters
# I instrument for both price and the within-nest share variables
# I create interaction terms between these instruments and the nest indicators
# This is to provide sufficient instruments for all endogenous variables
nested_logit_iv <- ivreg(ln_share_ratio ~ satellite + p + 
                           ln_within_nest_share_satellite + 
                           ln_within_nest_share_wired | 
                           satellite + x + w + 
                           I(x*satellite) + I(x*(1-satellite)) + 
                           I(w*satellite) + I(w*(1-satellite)), 
                         data = nested_data)

# View results
summary(nested_logit_iv)


# Yields a positive coefficient for price -- likely bad instruments for price.



# Option 2: use only the cost-shifters as instruments
cost_only_iv <- ivreg(ln_share_ratio ~ satellite + p + 
                        ln_within_nest_share_satellite + 
                        ln_within_nest_share_wired | 
                        satellite + w + 
                        I(w*satellite) + I(w*(1-satellite)), 
                      data = nested_data)

summary(cost_only_iv, diagnostics = TRUE)

# Price coefficient is negative, so that is a good indication.





# Approach 3: Create BLP-style instruments (characteristics of other products)
# These are based on the characteristics of competing products in the same market
blp_data <- market_product_data %>%
  # First create the base nested data structure
  group_by(market) %>%
  mutate(
    market_total_share = sum(s),
    outside_share = 1 - market_total_share
  ) %>%
  group_by(market, satellite) %>%
  mutate(
    nest_total_share = sum(s),
    within_nest_share = s / nest_total_share
  ) %>%
  ungroup() %>%
  mutate(
    ln_share_ratio = log(s / outside_share),
    ln_within_nest_share = log(within_nest_share),
    ln_within_nest_share_satellite = ln_within_nest_share * satellite,
    ln_within_nest_share_wired = ln_within_nest_share * (1-satellite)
  ) %>%
  # Now create BLP instruments
  group_by(market, satellite) %>%
  mutate(
    # Sum of x of other products in same nest
    sum_other_x_nest = sum(x) - x,
    # Sum of w of other products in same nest
    sum_other_w_nest = sum(w) - w
  ) %>%
  group_by(market) %>%
  mutate(
    # Sum of x of products in other nest
    sum_x_other_nest = sum(x * (1-satellite)) * satellite + sum(x * satellite) * (1-satellite),
    # Sum of w of products in other nest
    sum_w_other_nest = sum(w * (1-satellite)) * satellite + sum(w * satellite) * (1-satellite)
  ) %>%
  ungroup()

# BLP-style IV regression
blp_iv <- ivreg(ln_share_ratio ~ satellite + p + 
                  ln_within_nest_share_satellite + 
                  ln_within_nest_share_wired | 
                  satellite + w + sum_other_x_nest + sum_other_w_nest + 
                  sum_x_other_nest + sum_w_other_nest, 
                data = blp_data)

summary(blp_iv, diagnostics = TRUE)


# Price coefficient also negative




# Approach 4: Hausman-type instruments (prices in other markets)
hausman_data <- market_product_data %>%
  # Calculate average price of same product in other markets
  group_by(product) %>%
  mutate(avg_price_other_markets = (sum(p) - p)/(n() - 1)) %>%
  # Then create the usual nested structure
  group_by(market) %>%
  mutate(
    market_total_share = sum(s),
    outside_share = 1 - market_total_share
  ) %>%
  group_by(market, satellite) %>%
  mutate(
    nest_total_share = sum(s),
    within_nest_share = s / nest_total_share
  ) %>%
  ungroup() %>%
  mutate(
    ln_share_ratio = log(s / outside_share),
    ln_within_nest_share = log(within_nest_share),
    ln_within_nest_share_satellite = ln_within_nest_share * satellite,
    ln_within_nest_share_wired = ln_within_nest_share * (1-satellite)
  )

# Hausman IV regression
hausman_iv <- ivreg(ln_share_ratio ~ satellite + p + 
                      ln_within_nest_share_satellite + 
                      ln_within_nest_share_wired | 
                      satellite + avg_price_other_markets + w, 
                    data = hausman_data)

summary(hausman_iv, diagnostics = TRUE)

# Not enough instruments here, so this is not good. 



# Either the cost-shifter-only or the BLP estimation seem to work well at instrumenting for price.










## Question iv) Compile a table comparing estimated vs true own-price elasticities 

# OLS Multinomial Logit
ols_logit <- lm(ln_share_ratio ~ p + x + w + satellite, data = ols_logit_data)
summary(ols_logit)

# 2SLS Multinomial Logit
iv_logit <- ivreg(ln_share_ratio ~ satellite + p | satellite + x + w, data = logit_data)

# BLP Nested Logit (using BLP instruments as defined previously)
blp_nested_logit <- ivreg(ln_share_ratio ~ satellite + p + 
                            ln_within_nest_share_satellite + 
                            ln_within_nest_share_wired | 
                            satellite + w + sum_other_x_nest + sum_other_w_nest + 
                            sum_x_other_nest + sum_w_other_nest, 
                          data = blp_data)

# Step 2: Extract coefficients
# ---------------------------------------------------------------------------------
coef_ols <- coef(ols_logit)
coef_iv <- coef(iv_logit)
coef_blp <- coef(blp_nested_logit)

# Create a data frame to store coefficients across models
coefficients_df <- data.frame(
  Parameter = c("Intercept", "Satellite", "Price", "Nest Parameter (Satellite)", "Nest Parameter (Wired)"),
  OLS_Logit = c(coef_ols["(Intercept)"], coef_ols["satellite"], coef_ols["p"], NA, NA),
  IV_Logit = c(coef_iv["(Intercept)"], coef_iv["satellite"], coef_iv["p"], NA, NA),
  BLP_Nested = c(coef_blp["(Intercept)"], coef_blp["satellite"], coef_blp["p"], 
                 1 - coef_blp["ln_within_nest_share_satellite"], 
                 1 - coef_blp["ln_within_nest_share_wired"])
)

# Step 3: Calculate own-price elasticities for each model and product
# ---------------------------------------------------------------------------------
calculate_elasticities <- function(data, alpha_ols, alpha_iv, alpha_blp, 
                                   sigma_sat, sigma_wired) {
  # For each product in each market, calculate elasticities
  elasticities <- data %>%
    group_by(market, product) %>%
    mutate(
      # Multinomial Logit OLS
      elas_ols = alpha_ols * p * (1 - s),
      
      # Multinomial Logit IV
      elas_iv = alpha_iv * p * (1 - s),
      
      # Nested Logit BLP
      elas_blp = case_when(
        satellite == 1 ~ alpha_blp * p * (1 - sigma_sat * within_nest_share - (1 - sigma_sat) * s),
        satellite == 0 ~ alpha_blp * p * (1 - sigma_wired * within_nest_share - (1 - sigma_wired) * s)
      )
    ) %>%
    ungroup()
  
  return(elasticities)
}

# Calculate elasticities using the estimated parameters
elasticity_data <- calculate_elasticities(
  data = blp_data,  # Use the dataset with all necessary variables
  alpha_ols = coef_ols["p"],
  alpha_iv = coef_iv["p"],
  alpha_blp = coef_blp["p"],
  sigma_sat = 1 - coef_blp["ln_within_nest_share_satellite"],
  sigma_wired = 1 - coef_blp["ln_within_nest_share_wired"]
)

# Step 4: Calculate true own-price elasticities (assuming you have a demand function specified)
# ---------------------------------------------------------------------------------
# This is a placeholder - you need to replace this with the true elasticity calculation
# based on your specific demand function and parameters
calculate_true_elasticities <- function(data) {
  # Placeholder function - replace with your actual formula
  # For example, if you know the true parameters:
  true_alpha = -1.5  # Replace with your true value
  true_sigma_sat = 0.7  # Replace with your true value
  true_sigma_wired = 0.8  # Replace with your true value
  
  data %>%
    mutate(
      true_elasticity = case_when(
        satellite == 1 ~ true_alpha * p * (1 - true_sigma_sat * within_nest_share - (1 - true_sigma_sat) * s),
        satellite == 0 ~ true_alpha * p * (1 - true_sigma_wired * within_nest_share - (1 - true_sigma_wired) * s)
      )
    )
}

# Add true elasticities to the data
elasticity_data <- calculate_true_elasticities(elasticity_data)

# Step 5: Create summary statistics for elasticities
# ---------------------------------------------------------------------------------
elasticity_summary <- elasticity_data %>%
  group_by(satellite) %>%
  summarize(
    Type = if_else(first(satellite) == 1, "Satellite", "Wired"),
    Mean_OLS_Elasticity = mean(elas_ols),
    Mean_IV_Elasticity = mean(elas_iv),
    Mean_BLP_Elasticity = mean(elas_blp),
    Mean_True_Elasticity = mean(true_elasticity),
    RMSE_OLS = sqrt(mean((elas_ols - true_elasticity)^2)),
    RMSE_IV = sqrt(mean((elas_iv - true_elasticity)^2)),
    RMSE_BLP = sqrt(mean((elas_blp - true_elasticity)^2))
  ) %>%
  select(-satellite)

# Also calculate overall statistics
overall_elasticity <- elasticity_data %>%
  summarize(
    Type = "Overall",
    Mean_OLS_Elasticity = mean(elas_ols),
    Mean_IV_Elasticity = mean(elas_iv),
    Mean_BLP_Elasticity = mean(elas_blp),
    Mean_True_Elasticity = mean(true_elasticity),
    RMSE_OLS = sqrt(mean((elas_ols - true_elasticity)^2)),
    RMSE_IV = sqrt(mean((elas_iv - true_elasticity)^2)),
    RMSE_BLP = sqrt(mean((elas_blp - true_elasticity)^2))
  )

# Combine the summary tables
elasticity_summary <- bind_rows(elasticity_summary, overall_elasticity)

# Print the tables
kable(coefficients_df, digits = 4, caption = "Model Parameter Estimates")
kable(elasticity_summary, digits = 4, caption = "Own-Price Elasticity Comparison")






# Combined LaTex table

# Function to create a LaTeX table in landscape mode with title on top
create_latex_table <- function(data, filename, caption) {
  # Create LaTeX code for the table
  latex_table <- xtable(data, 
                        caption = caption,
                        digits = 4)
  
  # Define LaTeX document with the table in landscape mode
  cat("\\documentclass{article}
\\usepackage{booktabs}
\\usepackage{geometry}
\\usepackage{pdflscape}  % For landscape orientation
\\geometry{margin=1in}
\\begin{document}

\\begin{landscape}  % Start landscape environment
\\begin{table}[htbp]
\\caption{", caption, "}  % Title on top
\\centering
", file = filename)
  
  # Append the table to the document
  print(latex_table, 
        file = filename, 
        append = TRUE,
        include.rownames = FALSE,  # Set to TRUE if you want row numbers
        floating = FALSE,  # Already in a table environment
        table.placement = NULL,
        booktabs = TRUE,
        caption.placement = NULL)  # We're handling caption manually
  
  # Close the LaTeX document
  cat("\\end{table}
\\end{landscape}  % End landscape environment
\\end{document}", file = filename, append = TRUE)
}

# Create the coefficients table
create_latex_table(
  data = coefficients_df,
  filename = "coefficients_table.tex",
  caption = "Model Parameter Estimates"
)

# Create the elasticity comparison table
create_latex_table(
  data = elasticity_summary,
  filename = "elasticity_table.tex",
  caption = "Own-Price Elasticity Comparison"
)






# From the table we see that the IV-logit has the lowest RMSE.
# This estimation produces the least difference between the true and modeled own-price elasticity.

