#!/usr/bin/env Rscript

library(tidyverse)

length_file <- commandArgs(TRUE)[1]
output_file <- commandArgs(TRUE)[2]

d_lengths <- read_tsv(length_file)

p <- d_lengths |>
  pivot_longer(cols = starts_with("length_"),
               names_to = "frag_len",
               names_prefix = "length_",
               values_to = "count") |>
  mutate(frag_len = as.numeric(frag_len)) %>%
  ggplot(aes(x=frag_len,
             y=count,
             col=!!sym(names(.)[1]))) +
  geom_line() +
  geom_vline(aes(xintercept=167), lwd=0.5, lty=3) +
  scale_x_continuous(name="Fragment length (bp)") +
  scale_y_continuous(name="Relative frequency") +
  scale_color_brewer(palette="Set2") +
  theme_bw() +
  theme(legend.position=c(0.8, 0.6)) +
  guides(colour = guide_legend(override.aes = list(alpha=1, lwd=1))) +
  theme(legend.background = element_rect(size=0.2, linetype="solid", color="black"))

ggsave(output_file, p, width=10, height=6)
