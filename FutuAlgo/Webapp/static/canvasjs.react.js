var _createClass = function () { function defineProperties(target, props) { for (var i = 0; i < props.length; i++) { var descriptor = props[i]; descriptor.enumerable = descriptor.enumerable || false; descriptor.configurable = true; if ("value" in descriptor) descriptor.writable = true; Object.defineProperty(target, descriptor.key, descriptor); } } return function (Constructor, protoProps, staticProps) { if (protoProps) defineProperties(Constructor.prototype, protoProps); if (staticProps) defineProperties(Constructor, staticProps); return Constructor; }; }();

function _classCallCheck(instance, Constructor) { if (!(instance instanceof Constructor)) { throw new TypeError("Cannot call a class as a function"); } }

function _possibleConstructorReturn(self, call) { if (!self) { throw new ReferenceError("this hasn't been initialised - super() hasn't been called"); } return call && (typeof call === "object" || typeof call === "function") ? call : self; }

function _inherits(subClass, superClass) { if (typeof superClass !== "function" && superClass !== null) { throw new TypeError("Super expression must either be null or a function, not " + typeof superClass); } subClass.prototype = Object.create(superClass && superClass.prototype, { constructor: { value: subClass, enumerable: false, writable: true, configurable: true } }); if (superClass) Object.setPrototypeOf ? Object.setPrototypeOf(subClass, superClass) : subClass.__proto__ = superClass; }

CanvasJS = CanvasJS.Chart ? CanvasJS : window.CanvasJS;

var CanvasJSChart = function (_React$Component) {
	_inherits(CanvasJSChart, _React$Component);

	function CanvasJSChart(props) {
		_classCallCheck(this, CanvasJSChart);

		var _this = _possibleConstructorReturn(this, (CanvasJSChart.__proto__ || Object.getPrototypeOf(CanvasJSChart)).call(this, props));

		_this.options = props.options ? props.options : {};
		_this.containerProps = props.containerProps ? props.containerProps : { width: "100%", position: "relative" };
		_this.containerProps.height = props.containerProps && props.containerProps.height ? props.containerProps.height : _this.options.height ? _this.options.height + "px" : "400px";
		_this.chartContainerId = "canvasjs-react-chart-container-" + CanvasJSChart._cjsContainerId++;
		return _this;
	}

	_createClass(CanvasJSChart, [{
		key: 'componentDidMount',
		value: function componentDidMount() {
			//Create Chart and Render		
			this.chart = new CanvasJS.Chart(this.chartContainerId, this.options);
			this.chart.render();

			if (this.props.onRef) this.props.onRef(this.chart);
		}
	}, {
		key: 'shouldComponentUpdate',
		value: function shouldComponentUpdate(nextProps, nextState) {
			//Check if Chart-options has changed and determine if component has to be updated
			return !(nextProps.options === this.options);
		}
	}, {
		key: 'componentDidUpdate',
		value: function componentDidUpdate() {
			//Update Chart Options & Render
			this.chart.options = this.props.options;
			this.chart.render();
		}
	}, {
		key: 'componentWillUnmount',
		value: function componentWillUnmount() {
			//Destroy chart and remove reference
			this.chart.destroy();
			if (this.props.onRef) this.props.onRef(undefined);
		}
	}, {
		key: 'render',
		value: function render() {
			//return React.createElement('div', { id: this.chartContainerId, style: this.containerProps });		
			return React.createElement('div', { id: this.chartContainerId, style: this.containerProps });
		}
	}]);

	return CanvasJSChart;
}(React.Component);

CanvasJSChart._cjsContainerId = 0;


var CanvasJSReact = {
	CanvasJSChart: CanvasJSChart,
	CanvasJS: CanvasJS
};

