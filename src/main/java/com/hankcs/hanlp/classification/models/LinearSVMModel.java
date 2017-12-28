/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>me@hankcs.com</email>
 * <create-date>16/2/15 AM9:07</create-date>
 *
 * <copyright file="LinearSVMModel.java" company="码农场">
 * Copyright (c) 2008-2016, 码农场. All Right Reserved, http://www.hankcs.com/
 * This source is subject to Hankcs. Please contact Hankcs to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.classification.models;

import com.hankcs.hanlp.classification.features.IFeatureWeighter;
import de.bwaldvogel.liblinear.Model;

/**
 * 线性SVM模型
 *
 * @author hankcs
 */
public class LinearSVMModel extends AbstractModel
{
    /**
     * 训练样本数
     */
    public int n = 0;
    /**
     * 类别数
     */
    public int c = 0;
    /**
     * 特征数
     */
    public int d = 0;
    /**
     * 特征权重计算工具
     */
    public IFeatureWeighter featureWeighter;
    /**
     * SVM分类模型
     */
    public Model svmModel;
}