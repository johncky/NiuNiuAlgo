class SplineAreaChart extends React.Component{
    constructor(props){
        super(props);
    }
        
    render(){
        
        Xformat = this.props.Xlegend ? this.props.Xformat : "MM DD";
        unit = this.props.unit ? this.props.unit : "$";
        Xlegend = this.props.Xlegend ? this.props.Xlegend : "Datetime";
        Ylegend = this.props.Ylegend ? this.props.Ylegend : (unit);

        const options = {
            backgroundColor:"#f8f9fa",
            theme: 'light',
			animationEnabled: true,
			zoomEnabled: true,
			title: {
				text: this.props.title,
			},
			axisY: {
                gridThickness: 0,
                tickLength: 0,
                lineThickness: 0,
				title: Ylegend,
				includeZero: false,
				suffix: ""
			},
			data: [{
                color:'dark',
                lineThickness: 0,
                fillOpacity: 0.7,
                markerSize: 0,
				type: "area",
				xValueFormatString: Xformat,
				yValueFormatString: "#,##0.##",
				showInLegend: true,
				legendText: Xlegend,
				dataPoints: this.props.data.map(item => ({x: new Date(item.x), y: Number(item.y)})),
			}]
        }
        return (<div className='mt-5 mx-0'>
                    <CanvasJSChart options = {options}/>
                </div>);
    }
}


function ScrollableTable(props){
    return(
        <div className='scrollable_table'>
        <Table striped bordered hover size="sm">

            <thead>
                <tr>
                    {props.headers.map(h=><th key={h}>{h}</th>)}
                </tr>
            </thead>

            <tbody>
                {props.data.map((h, row_id)=><tr key={row_id}>{h.map((x, col_id)=><td key={row_id.toString()+col_id.toString()}>{x}</td>)}</tr>)}
            </tbody>
        </Table>
        </div>
    );
}