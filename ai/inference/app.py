#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
NLP-SecLogAI 推理API服务
提供日志解析、异常检测和安全报告生成等功能
"""

import os
from typing import List, Dict, Any, Optional
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from models.model_loader import load_classifier_model, load_anomaly_model
from api.nl2sql import NL2SQLConverter
from api.report_generator import ReportGenerator


# 初始化FastAPI应用
app = FastAPI(
    title="NLP-SecLogAI API",
    description="日志解析、异常检测和安全报告生成API",
    version="1.0.0"
)

# 加载模型
classifier_model = None
anomaly_model = None
nl2sql = None
report_generator = None


@app.on_event("startup")
async def startup_event():
    """服务启动时加载模型"""
    global classifier_model, anomaly_model, nl2sql, report_generator
    
    try:
        # 加载日志分类模型
        classifier_model = load_classifier_model(
            model_path=os.environ.get("CLASSIFIER_MODEL_PATH", "models/log_classifier_final.pt")
        )
        
        # 加载异常检测模型
        anomaly_model = load_anomaly_model(
            model_path=os.environ.get("ANOMALY_MODEL_PATH", "models/anomaly_detector_final.pt")
        )
        
        # 初始化NL2SQL转换器
        nl2sql = NL2SQLConverter()
        
        # 初始化报告生成器
        report_generator = ReportGenerator()
        
    except Exception as e:
        print(f"模型加载失败: {str(e)}")


# API请求/响应模型
class LogItem(BaseModel):
    """单条日志项"""
    log_text: str = Field(..., description="原始日志文本")
    source: Optional[str] = Field(None, description="日志来源（如'windows'、'firewall'等）")


class NLQueryRequest(BaseModel):
    """自然语言查询请求"""
    query: str = Field(..., description="自然语言查询文本")


class NLQueryResponse(BaseModel):
    """自然语言查询响应"""
    sql: str = Field(..., description="生成的SQL查询")
    confidence: float = Field(..., description="转换置信度")


class AnomalyDetectionRequest(BaseModel):
    """异常检测请求"""
    log_text: str = Field(..., description="要检测的日志文本")


class AnomalyDetectionResponse(BaseModel):
    """异常检测响应"""
    is_anomaly: bool = Field(..., description="是否为异常")
    score: float = Field(..., description="异常评分（0-1）")
    reason: Optional[str] = Field(None, description="异常原因")


class GenerateReportRequest(BaseModel):
    """生成报告请求"""
    logs: List[Dict[str, Any]] = Field(..., description="日志列表")
    time_range: Optional[str] = Field("24h", description="时间范围（如'24h'、'7d'）")


class GenerateReportResponse(BaseModel):
    """生成报告响应"""
    report: str = Field(..., description="生成的安全报告（Markdown格式）")


# API路由
@app.get("/")
async def root():
    """API根路由"""
    return {"message": "欢迎使用NLP-SecLogAI API"}


@app.post("/ai/nl2sql", response_model=NLQueryResponse)
async def convert_nl_to_sql(request: NLQueryRequest):
    """将自然语言转换为SQL查询"""
    if nl2sql is None:
        raise HTTPException(status_code=503, detail="NL2SQL服务未初始化")
    
    try:
        result = nl2sql.convert(request.query)
        return {
            "sql": result["sql"],
            "confidence": result["confidence"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"转换失败: {str(e)}")


@app.post("/ai/detect_anomaly", response_model=AnomalyDetectionResponse)
async def detect_anomaly(request: AnomalyDetectionRequest):
    """检测日志中的异常"""
    if anomaly_model is None:
        raise HTTPException(status_code=503, detail="异常检测模型未初始化")
    
    try:
        # 执行异常检测
        result = anomaly_model.predict(request.log_text)
        
        # 获取异常原因
        reason = None
        if result["is_anomaly"] and classifier_model is not None:
            # 使用分类模型获取日志类型
            log_type = classifier_model.predict(request.log_text)
            if log_type["label"] == 0:  # logon
                reason = "登录异常"
            elif log_type["label"] == 2:  # connection_blocked
                reason = "网络连接异常"
            else:
                reason = "未知异常模式"
        
        return {
            "is_anomaly": result["is_anomaly"],
            "score": result["score"],
            "reason": reason
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"异常检测失败: {str(e)}")


@app.post("/ai/generate_report", response_model=GenerateReportResponse)
async def generate_report(request: GenerateReportRequest):
    """生成安全报告"""
    if report_generator is None:
        raise HTTPException(status_code=503, detail="报告生成器未初始化")
    
    try:
        report = report_generator.generate(request.logs, request.time_range)
        return {"report": report}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"报告生成失败: {str(e)}")


if __name__ == "__main__":
    # 本地开发运行
    uvicorn.run("app:app", host="0.0.0.0", port=8001, reload=True) 