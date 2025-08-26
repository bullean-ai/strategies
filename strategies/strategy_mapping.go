package strategies

import (
	ffnnDomain "github.com/bullean-ai/bullean-go/neural_nets/domain"
	"github.com/bullean-ai/bullean-go/neural_nets/ffnn/layer/neuron/synapse"
	"github.com/bullean-ai/strategies/strategies/domain"
)

func GetStrategies() map[string]domain.IStrategyModel {
	return map[string]domain.IStrategyModel{
		"AIStrategyV1": NewAIStrategyV1(900, 80, 300, 0.0001, &ffnnDomain.Config{
			Inputs:     913,
			Layout:     []int{80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 3},
			Activation: ffnnDomain.ActivationReLU,
			Mode:       ffnnDomain.ModeRegression,
			Weight:     synapse.NewNormal(1e-20, 1e-20),
			Bias:       true,
		}),
	}
}
