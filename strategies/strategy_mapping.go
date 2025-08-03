package strategies

import "github.com/bullean-ai/strategies/strategies/domain"

var (
	STRATEGIES = map[string]domain.IStrategyModel{
		"AIStrategyV1": NewAIStrategyV1(10, 10, 10, 0.0001),
	}
)
