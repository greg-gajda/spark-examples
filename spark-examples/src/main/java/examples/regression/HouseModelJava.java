package examples.regression;

import java.sql.Date;
import java.text.ParseException;

import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;

import examples.common.LabeledPointConverter;

public class HouseModelJava implements LabeledPointConverter {

	private final Long id;
	private final Date date;
	private final Double price;
	private final Integer bedrooms;
	private final Double bathrooms;
	private final Integer sqft_living;
	private final Integer sqft_lot;
	private final Double floors;
	private final Integer waterfront;
	private final Integer view;
	private final Integer condition;
	private final Integer grade;
	private final Integer sqft_above;
	private final Integer sqft_basement;
	private final Integer yr_built;
	private final Integer yr_renovated;
	private final String zipcode;
	private final Double latitude;
	private final Double longitude;
	private final Integer sqft_living15;
	private final Integer sqft_lot15;

	public HouseModelJava(Long id, Date date, Double price, Integer bedrooms, Double bathrooms, Integer sqft_living,
			Integer sqft_lot, Double floors, Integer waterfront, Integer view, Integer condition, Integer grade,
			Integer sqft_above, Integer sqft_basement, Integer yr_built, Integer yr_renovated, String zipcode,
			Double latitude, Double longitude, Integer sqft_living15, Integer sqft_lot15) {
		super();
		this.id = id;
		this.date = date;
		this.price = price;
		this.bedrooms = bedrooms;
		this.bathrooms = bathrooms;
		this.sqft_living = sqft_living;
		this.sqft_lot = sqft_lot;
		this.floors = floors;
		this.waterfront = waterfront;
		this.view = view;
		this.condition = condition;
		this.grade = grade;
		this.sqft_above = sqft_above;
		this.sqft_basement = sqft_basement;
		this.yr_built = yr_built;
		this.yr_renovated = yr_renovated;
		this.zipcode = zipcode;
		this.latitude = latitude;
		this.longitude = longitude;
		this.sqft_living15 = sqft_living15;
		this.sqft_lot15 = sqft_lot15;
	}

	public HouseModelJava(String... row) {
		this(Long.parseLong(row[0]), new Date(parseDate(row[1])), Double.parseDouble(row[2]), Integer.parseInt(row[3]),
				Double.parseDouble(row[4]), Integer.parseInt(row[5]), Integer.parseInt(row[6]), Double
						.parseDouble(row[7]), Integer.parseInt(row[8]), Integer.parseInt(row[9]), Integer
						.parseInt(row[10]), Integer.parseInt(row[11]), Integer.parseInt(row[12]), Integer
						.parseInt(row[13]), Integer.parseInt(row[14]), Integer.parseInt(row[15]), row[16], Double
						.parseDouble(row[17]), Double.parseDouble(row[18]), Integer.parseInt(row[19]), Integer
						.parseInt(row[20]));
	}

	@Override
	public LabeledPoint toLabeledPoint() {
		return new LabeledPoint(label(), features());
	}

	@Override
	public double label() {
		return price;
	}

	@Override
	public Vector features() {
		double[] features = { id, bedrooms, bathrooms, sqft_living, sqft_lot, floors, waterfront, view, condition,
				grade, sqft_above, sqft_basement, yr_built, yr_renovated, latitude, longitude, sqft_living15,
				sqft_lot15 };
		return Vectors.dense(features);
	}

	static Long parseDate(String value) {
		try {
			return new java.text.SimpleDateFormat("yyyyMMdd'T'hhmmss").parse(value).getTime();
		} catch (ParseException e) {
			throw new RuntimeException(e);
		}
	}

	public Long getId() {
		return id;
	}

	public java.sql.Date getDate() {
		return date;
	}

	public Double getPrice() {
		return price;
	}

	public Integer getBedrooms() {
		return bedrooms;
	}

	public Double getBathrooms() {
		return bathrooms;
	}

	public Integer getSqft_living() {
		return sqft_living;
	}

	public Integer getSqft_lot() {
		return sqft_lot;
	}

	public Double getFloors() {
		return floors;
	}

	public Integer getWaterfront() {
		return waterfront;
	}

	public Integer getView() {
		return view;
	}

	public Integer getCondition() {
		return condition;
	}

	public Integer getGrade() {
		return grade;
	}

	public Integer getSqft_above() {
		return sqft_above;
	}

	public Integer getSqft_basement() {
		return sqft_basement;
	}

	public Integer getYr_built() {
		return yr_built;
	}

	public Integer getYr_renovated() {
		return yr_renovated;
	}

	public String getZipcode() {
		return zipcode;
	}

	public Double getLatitude() {
		return latitude;
	}

	public Double getLongitude() {
		return longitude;
	}

	public Integer getSqft_living15() {
		return sqft_living15;
	}

	public Integer getSqft_lot15() {
		return sqft_lot15;
	}

}
