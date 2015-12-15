/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package examples.classification;

import java.util.HashMap;
import java.util.Map;

import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.regression.LabeledPoint;

public class BikeBuyerModelJava implements LabeledPointConverter {
	
	private final Integer customerKey;
	private final Integer age;
	private final Integer bikeBuyer;
	private final String commuteDistance;
	private final String englishEducation;
	private final String gender;
	private final Integer houseOwnerFlag;
	private final String maritalStatus;
	private final Integer numberCarsOwned;
	private final Integer numberChildrenAtHome;
	private final String englishOccupation;
	private final String region;
	private final Integer totalChildren;
	private final Float yearlyIncome;

	public BikeBuyerModelJava(Integer customerKey, Integer age, Integer bikeBuyer, String commuteDistance,
			String englishEducation, String gender, Integer houseOwnerFlag, String maritalStatus,
			Integer numberCarsOwned, Integer numberChildrenAtHome, String englishOccupation, String region,
			Integer totalChildren, Float yearlyIncome) {
		super();
		this.customerKey = customerKey;
		this.age = age;
		this.bikeBuyer = bikeBuyer;
		this.commuteDistance = commuteDistance;
		this.englishEducation = englishEducation;
		this.gender = gender;
		this.houseOwnerFlag = houseOwnerFlag;
		this.maritalStatus = maritalStatus;
		this.numberCarsOwned = numberCarsOwned;
		this.numberChildrenAtHome = numberChildrenAtHome;
		this.englishOccupation = englishOccupation;
		this.region = region;
		this.totalChildren = totalChildren;
		this.yearlyIncome = yearlyIncome;
	}

	public BikeBuyerModelJava(String... row) {
		this(Integer.valueOf(row[0]), Integer.valueOf(row[1]), Integer.valueOf(row[2]), row[3], row[4], row[5], Integer
				.valueOf(row[6]), row[7], Integer.valueOf(row[8]), Integer.valueOf(row[9]), row[10], row[11], Integer
				.valueOf(row[12]), Float.valueOf(row[13].replaceFirst(",", ".")));
	}

	@Override
	public LabeledPoint toLabeledPoint() {
		return new LabeledPoint(label(), features());
	}
	
	@Override
	public double label() {
		return bikeBuyer.doubleValue();
	}

	@Override
	public Vector features() {
		double[] features = new double[getClass().getDeclaredFields().length - 1];
		features[0] = customerKey.doubleValue();
		features[1] = age.doubleValue();
		switch (commuteDistance) {
		case "0-1 Miles":
			features[2] = 0d;
			break;
		case "1-2 Miles":
			features[2] = 1d;
			break;
		case "2-5 Miles":
			features[2] = 2d;
			break;
		case "5-10 Miles":
			features[2] = 3d;
			break;
		case "10+ Miles":
			features[2] = 4d;
			break;
		default:
		}
		switch (englishEducation) {
		case "High School":
			features[3] = 0d;
			break;
		case "Partial High School":
			features[3] = 1d;
			break;
		case "Partial College":
			features[3] = 2d;
			break;
		case "Graduate Degree":
			features[3] = 3d;
			break;
		case "Bachelors":
			features[3] = 4d;
			break;
		default:
		}
		switch (gender) {
		case "M":
			features[4] = 0d;
			break;
		case "F":
			features[4] = 1d;
			break;
		default:
		}
		features[5] = houseOwnerFlag.doubleValue();
		switch (maritalStatus) {
		case "S":
			features[6] = 0d;
			break;
		case "M":
			features[6] = 1d;
			break;
		default:
		}
		features[7] = numberCarsOwned.doubleValue();
		features[8] = numberChildrenAtHome.doubleValue();
		switch (englishOccupation) {
		case "Professional":
			features[9] = 0d;
			break;
		case "Clerical":
			features[9] = 1d;
			break;
		case "Manual":
			features[9] = 2d;
			break;
		case "Management":
			features[9] = 3d;
			break;
		case "Skilled Manual":
			features[9] = 4d;
			break;
		default:
		}
		switch (region) {
		case "North America":
			features[10] = 0d;
			break;
		case "Pacific":
			features[10] = 1d;
			break;
		case "Europe":
			features[10] = 2d;
			break;
		default:
		}
		features[11] = totalChildren.doubleValue();
		features[12] = yearlyIncome;
		return Vectors.dense(features);
	}

	public static Map<Integer, Integer> categoricalFeaturesInfo() {
		return new HashMap<Integer, Integer>() {
			private static final long serialVersionUID = 1L;
			{
				put(2, 5);
				put(3, 5);
				put(4, 2);
				put(6, 2);
				put(9, 5);
				put(10, 3);
			}
		};
	}

	public Integer getCustomerKey() {
		return customerKey;
	}

	public Integer getAge() {
		return age;
	}

	public Integer getBikeBuyer() {
		return bikeBuyer;
	}

	public String getCommuteDistance() {
		return commuteDistance;
	}

	public String getEnglishEducation() {
		return englishEducation;
	}

	public String getGender() {
		return gender;
	}

	public Integer getHouseOwnerFlag() {
		return houseOwnerFlag;
	}

	public String getMaritalStatus() {
		return maritalStatus;
	}

	public Integer getNumberCarsOwned() {
		return numberCarsOwned;
	}

	public Integer getNumberChildrenAtHome() {
		return numberChildrenAtHome;
	}

	public String getEnglishOccupation() {
		return englishOccupation;
	}

	public String getRegion() {
		return region;
	}

	public Integer getTotalChildren() {
		return totalChildren;
	}

	public Float getYearlyIncome() {
		return yearlyIncome;
	}

}
