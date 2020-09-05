CREATE TABLE IF NOT EXISTS `FUTU_K_DAY` (
  `ticker` VARCHAR(40) NOT NULL,
  `datetime` DATETIME NOT NULL,
  `open` FLOAT NULL,
  `high` FLOAT NULL,
  `low` FLOAT NULL,
  `close` FLOAT NULL,
  `volume` FLOAT NULL,
  `turnover` FLOAT NULL,
  PRIMARY KEY (`ticker`, `datetime`));

CREATE TABLE IF NOT EXISTS `FUTU_K_1M` (
  `ticker` VARCHAR(40) NOT NULL,
  `datetime` DATETIME NOT NULL,
  `open` FLOAT NULL,
  `high` FLOAT NULL,
  `low` FLOAT NULL,
  `close` FLOAT NULL,
  `volume` FLOAT NULL,
  `turnover` FLOAT NULL,
  PRIMARY KEY (`ticker`, `datetime`));

CREATE TABLE IF NOT EXISTS `FUTU_K_3M` (
  `ticker` VARCHAR(40) NOT NULL,
  `datetime` DATETIME NOT NULL,
  `open` FLOAT NULL,
  `high` FLOAT NULL,
  `low` FLOAT NULL,
  `close` FLOAT NULL,
  `volume` FLOAT NULL,
  `turnover` FLOAT NULL,
  PRIMARY KEY (`ticker`, `datetime`));

CREATE TABLE IF NOT EXISTS `FUTU_K_5M` (
  `ticker` VARCHAR(40) NOT NULL,
  `datetime` DATETIME NOT NULL,
  `open` FLOAT NULL,
  `high` FLOAT NULL,
  `low` FLOAT NULL,
  `close` FLOAT NULL,
  `volume` FLOAT NULL,
  `turnover` FLOAT NULL,
  PRIMARY KEY (`ticker`, `datetime`));

CREATE TABLE IF NOT EXISTS `FUTU_K_15M` (
  `ticker` VARCHAR(40) NOT NULL,
  `datetime` DATETIME NOT NULL,
  `open` FLOAT NULL,
  `high` FLOAT NULL,
  `low` FLOAT NULL,
  `close` FLOAT NULL,
  `volume` FLOAT NULL,
  `turnover` FLOAT NULL,
  PRIMARY KEY (`ticker`, `datetime`));


CREATE TABLE IF NOT EXISTS `FUTU_ORDER_UPDATE` (
  `trd_side` VARCHAR(10) NULL,
  `order_type` VARCHAR(20) NULL,
  `order_id` VARCHAR(50) NOT NULL,
  `ticker` VARCHAR(40) NULL,
  `stock_name` VARCHAR(50) NULL,
  `qty` FLOAT NULL,
  `price` FLOAT NULL,
  `create_time` DATETIME NULL,
  `updated_time` DATETIME NULL,
  `dealt_qty` FLOAT NULL,
  `dealt_avg_price` FLOAT NULL,
  `trd_env` VARCHAR(40) NULL,
  `order_status` VARCHAR(40) NULL,
  `trd_market` VARCHAR(40) NULL,
  `last_err_msg` VARCHAR(200) NULL,
  `remark` VARCHAR(200) NULL,
  PRIMARY KEY (`order_id`));


CREATE TABLE IF NOT EXISTS `FUTU_QUOTE` (
 `ticker` VARCHAR(40) NOT NULL,
 `datetime` DATETIME NOT NULL,
 `quote` FLOAT NULL,
 `volume` FLOAT NULL,
 PRIMARY KEY (`ticker`, `datetime`));