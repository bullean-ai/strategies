package main

import (
	domain2 "github.com/bullean-ai/bullean-go/binance/domain"
	"github.com/bullean-ai/bullean-go/data/domain"
)

type IStrategyModel interface {
	Init([]domain.Candle, func(string, bool))
	OnCandle(domain.Candle)
	UpdateBinanceClients([]domain2.IBinanceClient)
}
