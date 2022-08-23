library(tsibble)
library(dplyr)
library(fpp3)
tourism <- tsibble::tourism %>%
  mutate(State = recode(State,
                        `New South Wales` = "NSW",
                        `Northern Territory` = "NT",
                        `Queensland` = "QLD",
                        `South Australia` = "SA",
                        `Tasmania` = "TAS",
                        `Victoria` = "VIC",
                        `Western Australia` = "WA"
  ))

tourism_hts <- tourism %>%
  aggregate_key(State / Region, Trips = sum(Trips))

tourism_full <- tourism %>%
  aggregate_key((State/Region) * Purpose, Trips = sum(Trips))

fit <- tourism_full %>%
  filter(year(Quarter) <= 2015) %>%
  model(base = ETS(Trips)) %>%
  reconcile(
    bu = bottom_up(base),
    ols = min_trace(base, method = "ols"),
    mint = min_trace(base, method = "mint_shrink"),
  )

fc <- fit %>% forecast(h = "2 years")

fc %>%
  filter(is_aggregated(Region), is_aggregated(Purpose)) %>%
  autoplot(
    tourism_full %>% filter(year(Quarter) >= 2011),
    level = NULL
  ) +
  labs(y = "Trips ('000)") +
  facet_wrap(vars(State), scales = "free_y")

fc %>%
  filter(is_aggregated(State), is_aggregated(Purpose)) %>%
  accuracy(
    data = tourism_full,
    measures = list(rmse = RMSE, mase = MASE)
  ) %>%
  group_by(.model) %>%
  summarise(rmse = mean(rmse), mase = mean(mase))