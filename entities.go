package main

var (
	STRATEGIES = map[string]IStrategyModel{
		"AIStrategyV1": NewAIStrategyV1(10, 10, 10, 0.0001),
	}
)
