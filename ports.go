package main

import (
	binanceDomain "github.com/bullean-ai/bullean-go/binance/domain"
	"github.com/bullean-ai/bullean-go/data/domain"
)

type IStrategyModel interface {
	Init(string, string, string, []binanceDomain.IBinanceClient, []domain.Candle, func(string, bool))
	OnCandle(domain.Candle)
	UpdateBinanceClients([]binanceDomain.IBinanceClient)
}
