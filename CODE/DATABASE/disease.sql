/*
SQLyog Enterprise - MySQL GUI v6.56
MySQL - 5.5.5-10.1.13-MariaDB : Database - disease
*********************************************************************
*/

/*!40101 SET NAMES utf8 */;

/*!40101 SET SQL_MODE=''*/;

/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;

CREATE DATABASE /*!32312 IF NOT EXISTS*/`disease` /*!40100 DEFAULT CHARACTER SET latin1 */;

USE `disease`;

/*Table structure for table `doctorreg` */

DROP TABLE IF EXISTS `doctorreg`;

CREATE TABLE `doctorreg` (
  `Name` varchar(20) NOT NULL,
  `Email` varchar(20) NOT NULL,
  `Password` varchar(20) NOT NULL,
  `Confirm_password` varchar(20) NOT NULL,
  `OTP` int(4) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

/*Data for the table `doctorreg` */

insert  into `doctorreg`(`Name`,`Email`,`Password`,`Confirm_password`,`OTP`) values ('q','q@gmail.com','123','',0),('tony','tonystark17695@gmail','Gaurav17','',941),('tony','tonystark17695@gmail','Gaurav17','',941),('tony','tonystark17695@gmail','Gaurav17','',941),('tony','tonystark17695@gmail','Gaurav17','',4951),('tony','tonystark17695@gmail','Gaurav17','',7248);

/*Table structure for table `patientreg` */

DROP TABLE IF EXISTS `patientreg`;

CREATE TABLE `patientreg` (
  `Name` varchar(20) NOT NULL,
  `Age` int(20) NOT NULL,
  `Email` varchar(20) NOT NULL,
  `Password` varchar(20) NOT NULL,
  `Confirm_Password` varchar(20) NOT NULL,
  `OTP` int(4) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

/*Data for the table `patientreg` */

insert  into `patientreg`(`Name`,`Age`,`Email`,`Password`,`Confirm_Password`,`OTP`) values ('a',27,'a@gmail.com','123','',0),('tony',25,'tonystark17695@gmail','Gaurav17','',6522),('tony',27,'tonystark17695@gmail','Gaurav17','',5585),('tony',27,'tonystark17695@gmail','Gaurav17','',7673),('tony',27,'tonystark17695@gmail','Gaurav17','',3397),('tony',27,'tonystark17695@gmail','Gaurav17','',72),('tony',27,'tonystark17695@gmail','Gaurav17','',4066),('tony',27,'tonystark17695@gmail','Gaurav17','',4066),('tony',27,'tonystark17695@gmail','Gaurav17','',8385),('tony',27,'tonystark17695@gmail','Gaurav17','',8385);

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
