package strategies

import "github.com/bullean-ai/strategies/strategies/domain"

func GetStrategies() map[string]domain.IStrategyModel {
	return map[string]domain.IStrategyModel{
		"AIStrategyV1": NewAIStrategyV1(800, 80, 500, 0.00005),
	}
}
